from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import Field

from .models import CallEvent, FrozenModel, Reservation, Usage


class BudgetTier(FrozenModel):
    name: str
    token_limit: int = Field(gt=0)
    retrieval_limit: int = Field(gt=0)
    planned_query_limit: int = Field(gt=0)
    candidate_limit: int = Field(gt=0)
    repair_limit: int = Field(ge=0)


BUDGET_TIERS: Mapping[str, BudgetTier] = MappingProxyType(
    {
        "low": BudgetTier(
            name="low",
            token_limit=4096,
            retrieval_limit=2,
            planned_query_limit=1,
            candidate_limit=1,
            repair_limit=0,
        ),
        "middle": BudgetTier(
            name="middle",
            token_limit=12288,
            retrieval_limit=6,
            planned_query_limit=2,
            candidate_limit=2,
            repair_limit=1,
        ),
        "high": BudgetTier(
            name="high",
            token_limit=32768,
            retrieval_limit=12,
            planned_query_limit=4,
            candidate_limit=4,
            repair_limit=1,
        ),
    }
)


class BudgetExceeded(RuntimeError):
    pass


class ProtocolViolation(RuntimeError):
    def __init__(self, codes: tuple[str, ...], event: CallEvent) -> None:
        self.codes = codes
        self.event = event
        super().__init__(f"budget protocol violation: {', '.join(codes)}")


class BudgetLedger:
    def __init__(self, tier: BudgetTier) -> None:
        self.tier = tier
        self._next_reservation = 1
        self._active: dict[str, Reservation] = {}
        self._stages: dict[str, str] = {}
        self._events: list[CallEvent] = []
        self._spent_tokens = 0
        self._protocol_violations: list[str] = []

    @property
    def spent_tokens(self) -> int:
        return self._spent_tokens

    @property
    def reserved_tokens(self) -> int:
        return sum(reservation.reserved_tokens for reservation in self._active.values())

    @property
    def available_tokens(self) -> int:
        return self.tier.token_limit - self.spent_tokens - self.reserved_tokens

    @property
    def events(self) -> tuple[CallEvent, ...]:
        return tuple(self._events)

    @property
    def protocol_violations(self) -> tuple[str, ...]:
        return tuple(self._protocol_violations)

    def authorize(
        self,
        *,
        stage: str,
        prompt_tokens: int,
        max_output_tokens: int,
    ) -> Reservation:
        if prompt_tokens < 0 or max_output_tokens < 0:
            raise ValueError("reservation token counts must be nonnegative")
        requested = prompt_tokens + max_output_tokens
        if requested > self.available_tokens:
            raise BudgetExceeded(
                f"call requires {requested} tokens with only {self.available_tokens} available"
            )
        reservation_id = f"{self.tier.name}-r{self._next_reservation}"
        self._next_reservation += 1
        reservation = Reservation(
            reservation_id=reservation_id,
            prompt_tokens=prompt_tokens,
            max_output_tokens=max_output_tokens,
        )
        self._active[reservation_id] = reservation
        self._stages[reservation_id] = stage
        return reservation

    def commit(self, reservation: Reservation, usage: Usage) -> CallEvent:
        active = self._active.get(reservation.reservation_id)
        if active is not reservation:
            raise ValueError("reservation is not active in this ledger")
        if usage.prompt_tokens is None or usage.completion_tokens is None:
            raise ValueError("authoritative prompt and completion usage is required")

        violation_codes: list[str] = []
        if usage.prompt_tokens != reservation.prompt_tokens:
            violation_codes.append("prompt_token_mismatch")
        if usage.completion_tokens > reservation.max_output_tokens:
            violation_codes.append("output_reservation_overrun")

        del self._active[reservation.reservation_id]
        stage = self._stages.pop(reservation.reservation_id)
        self._spent_tokens += usage.prompt_tokens + usage.completion_tokens
        if self._spent_tokens > self.tier.token_limit:
            violation_codes.append("hard_cap_overrun")
        event = CallEvent(
            stage=stage,
            reservation=reservation,
            usage=usage,
            protocol_violation=bool(violation_codes),
        )
        self._events.append(event)
        if violation_codes:
            self._protocol_violations.extend(violation_codes)
            raise ProtocolViolation(tuple(violation_codes), event)
        return event
