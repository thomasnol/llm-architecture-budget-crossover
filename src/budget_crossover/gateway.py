from __future__ import annotations

import asyncio
import fnmatch
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from .models import GatewayResponse, Usage

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
CONGESTION_STATUS_CODES = {408, 425, 429, 502, 503, 504}


def _truthy(value: str | None) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content or "")


class CubicConcurrencyLimiter:
    """CUBIC-inspired adaptive concurrency window for one credential.

    RFC 9438 controls packets, while this controller treats one completed HTTP
    request as an acknowledgement and one rate-limit/overload response as a
    congestion event. Virtual time advances by 1/cwnd per successful request,
    so roughly one full concurrency window corresponds to one CUBIC round.
    """

    def __init__(
        self,
        max_concurrency: int,
        *,
        initial_concurrency: int | None = None,
        beta: float = 0.7,
        cubic_c: float = 0.4,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if not 0 < beta < 1:
            raise ValueError("beta must be between 0 and 1")
        if cubic_c <= 0:
            raise ValueError("cubic_c must be positive")
        initial = initial_concurrency or min(4, max_concurrency)
        initial = max(1, min(initial, max_concurrency))
        self.max_concurrency = max_concurrency
        self.beta = beta
        self.cubic_c = cubic_c
        self._window = float(initial)
        self._w_max = float(initial)
        self._epoch_window = float(initial)
        self._virtual_time = 0.0
        self._k = 0.0
        self._active = 0
        self._condition = asyncio.Condition()

    @property
    def window(self) -> float:
        return self._window

    @property
    def limit(self) -> int:
        return max(1, min(self.max_concurrency, math.floor(self._window + 1e-9)))

    @property
    def active(self) -> int:
        return self._active

    @property
    def available(self) -> int:
        return max(0, self.limit - self._active)

    async def acquire(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._active < self.limit)
            self._active += 1

    async def release(self, *, successful: bool = False, congestion: bool = False) -> None:
        if successful and congestion:
            raise ValueError("a request cannot be both successful and congested")
        async with self._condition:
            if self._active < 1:
                raise RuntimeError("release called without a matching acquire")
            self._active -= 1
            if congestion:
                self._on_congestion()
            elif successful:
                self._on_success()
            self._condition.notify_all()

    def _on_congestion(self) -> None:
        current = self._window
        # RFC 9438 fast convergence: lower the remembered saturation point
        # when a new congestion event occurs below the previous W_max.
        if current < self._w_max:
            self._w_max = current * (1 + self.beta) / 2
        else:
            self._w_max = current
        reduced = max(1.0, current * self.beta)
        self._window = min(float(self.max_concurrency), reduced)
        self._epoch_window = self._window
        self._virtual_time = 0.0
        self._k = max(
            0.0,
            ((self._w_max - self._window) / self.cubic_c) ** (1 / 3),
        )

    def _on_success(self) -> None:
        if self._window >= self.max_concurrency:
            self._window = float(self.max_concurrency)
            return
        self._virtual_time += 1.0 / max(1.0, self._window)
        cubic_target = (
            self.cubic_c * (self._virtual_time - self._k) ** 3 + self._w_max
        )
        alpha = 3 * (1 - self.beta) / (1 + self.beta)
        reno_target = self._epoch_window + alpha * self._virtual_time
        target = min(float(self.max_concurrency), max(cubic_target, reno_target))
        if target > self._window:
            # One request completion is one ACK analogue. Cap each ACK-sized
            # increment to avoid a large jump after an application-limited gap.
            self._window += min(target - self._window, 1.0 / self._window)
            self._window = min(float(self.max_concurrency), self._window)


@dataclass
class CredentialSlot:
    index: int
    concurrency: int
    model_patterns: tuple[str, ...]
    api_key: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    limiter: CubicConcurrencyLimiter = field(init=False)
    access_token: str | None = None
    access_token_expires_at: float = 0.0
    token_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.limiter = CubicConcurrencyLimiter(self.concurrency)

    def supports(self, model: str) -> bool:
        return any(
            pattern == "*" or fnmatch.fnmatch(model, pattern) for pattern in self.model_patterns
        )


class GatewayClient:
    def __init__(self, timeout_seconds: float = 180.0) -> None:
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
        self.base_url = os.getenv("LLM_GATEWAY_BASE_URL", "").rstrip("/")
        self.token_url = os.getenv("LLM_GATEWAY_TOKEN_URL", "")
        self.scope = os.getenv("LLM_GATEWAY_SCOPE", "")
        self.chat_path = os.getenv("LLM_GATEWAY_CHAT_PATH", "/chat/completions")
        self.max_tokens_field = os.getenv("LLM_GATEWAY_MAX_TOKENS_FIELD", "max_tokens")
        self.api_key_header = os.getenv("LLM_GATEWAY_API_KEY_HEADER", "Authorization")
        self.api_key_prefix = os.getenv("LLM_GATEWAY_API_KEY_PREFIX", "Bearer ")
        self.oauth_basic_auth = _truthy(os.getenv("LLM_GATEWAY_OAUTH_BASIC_AUTH"))
        self.extra_headers = json.loads(os.getenv("LLM_GATEWAY_EXTRA_HEADERS", "{}"))
        per_key = int(os.getenv("LLM_GATEWAY_CONCURRENCY_PER_KEY", "4"))
        self.slots: list[CredentialSlot] = []
        default_models = {
            1: (
                "gpt-5.4,gpt-5.4-mini,gpt-5.4-nano,"
                "claude-opus-4-6,claude-sonnet-4-6"
            ),
            2: "gpt-5.4,gpt-5.4-mini,gpt-5.4-nano",
        }
        for index in (1, 2):
            api_key = os.getenv(f"LLM_GATEWAY_API_KEY_{index}") or None
            client_id = os.getenv(f"LLM_GATEWAY_CLIENT_ID_{index}") or None
            client_secret = os.getenv(f"LLM_GATEWAY_CLIENT_SECRET_{index}") or None
            if api_key or (client_id and client_secret):
                raw_patterns = os.getenv(
                    f"LLM_GATEWAY_MODELS_{index}", default_models[index]
                )
                patterns = tuple(part.strip() for part in raw_patterns.split(",") if part.strip())
                self.slots.append(
                    CredentialSlot(
                        index=index,
                        concurrency=per_key,
                        model_patterns=patterns,
                        api_key=api_key,
                        client_id=client_id,
                        client_secret=client_secret,
                    )
                )
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
        self._rr = 0
        self._rr_lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.slots)

    @property
    def maximum_total_concurrency(self) -> int:
        return sum(slot.concurrency for slot in self.slots)

    async def close(self) -> None:
        await self.client.aclose()

    async def _slot(self, model: str) -> CredentialSlot:
        eligible = [slot for slot in self.slots if slot.supports(model)]
        if not eligible:
            raise RuntimeError(f"no configured credential supports model {model!r}")
        async with self._rr_lock:
            start = self._rr % len(eligible)
            self._rr += 1
            rotated = eligible[start:] + eligible[:start]
            return max(
                rotated,
                key=lambda slot: (
                    slot.limiter.available,
                    -slot.limiter.active,
                ),
            )

    async def _oauth_token(self, slot: CredentialSlot) -> str:
        if slot.access_token and time.monotonic() < slot.access_token_expires_at:
            return slot.access_token
        if not self.token_url or not slot.client_id or not slot.client_secret:
            raise RuntimeError("OAuth2 credential is incomplete")
        async with slot.token_lock:
            if slot.access_token and time.monotonic() < slot.access_token_expires_at:
                return slot.access_token
            data = {"grant_type": "client_credentials"}
            if self.scope:
                data["scope"] = self.scope
            auth = None
            if self.oauth_basic_auth:
                auth = (slot.client_id, slot.client_secret)
            else:
                data["client_id"] = slot.client_id
                data["client_secret"] = slot.client_secret
            response = await self.client.post(self.token_url, data=data, auth=auth)
            response.raise_for_status()
            payload = response.json()
            slot.access_token = str(payload["access_token"])
            expires_in = max(60, int(payload.get("expires_in", 3600)))
            slot.access_token_expires_at = time.monotonic() + expires_in - 60
            return slot.access_token

    async def _headers(self, slot: CredentialSlot) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if slot.api_key:
            headers[self.api_key_header] = f"{self.api_key_prefix}{slot.api_key}"
        else:
            headers["Authorization"] = f"Bearer {await self._oauth_token(slot)}"
        return headers

    async def list_models(self) -> dict[str, Any]:
        if not self.configured:
            raise RuntimeError("gateway endpoint/credentials are not configured")
        slot = self.slots[0]
        await slot.limiter.acquire()
        try:
            response = await self.client.get(
                f"{self.base_url}/models",
                headers=await self._headers(slot),
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as error:
            await slot.limiter.release(
                congestion=error.response.status_code in CONGESTION_STATUS_CODES
            )
            raise
        except httpx.HTTPError:
            await slot.limiter.release(congestion=True)
            raise
        except Exception:
            await slot.limiter.release()
            raise
        else:
            await slot.limiter.release(successful=True)
            return payload

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> GatewayResponse:
        if not self.configured:
            raise RuntimeError("gateway endpoint/credentials are not configured")
        last_error: Exception | None = None
        for attempt in range(5):
            slot = await self._slot(model)
            await slot.limiter.acquire()
            try:
                started = time.perf_counter()
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    self.max_tokens_field: int(max_tokens),
                    "temperature": float(temperature),
                }
                response = await self.client.post(
                    f"{self.base_url}{self.chat_path}",
                    headers=await self._headers(slot),
                    json=payload,
                )
                latency = time.perf_counter() - started
                if response.status_code in RETRYABLE_STATUS_CODES:
                    response.raise_for_status()
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                usage_raw = body.get("usage", {})
                usage = Usage(
                    prompt_tokens=usage_raw.get("prompt_tokens", usage_raw.get("input_tokens")),
                    completion_tokens=usage_raw.get(
                        "completion_tokens", usage_raw.get("output_tokens")
                    ),
                    total_tokens=usage_raw.get("total_tokens"),
                )
                if usage.total_tokens is None and (
                    usage.prompt_tokens is not None and usage.completion_tokens is not None
                ):
                    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
                result = GatewayResponse(
                    text=_content_text(choice.get("message", {}).get("content", "")),
                    model=str(body.get("model", model)),
                    usage=usage,
                    latency_seconds=latency,
                    credential_slot=slot.index,
                    request_id=response.headers.get("x-request-id"),
                    raw_finish_reason=choice.get("finish_reason"),
                )
            except httpx.HTTPStatusError as error:
                await slot.limiter.release(
                    congestion=error.response.status_code in CONGESTION_STATUS_CODES
                )
                last_error = error
                if error.response.status_code not in RETRYABLE_STATUS_CODES:
                    raise
            except httpx.HTTPError as error:
                await slot.limiter.release(congestion=True)
                last_error = error
            except (KeyError, ValueError) as error:
                await slot.limiter.release()
                last_error = error
            except Exception:
                await slot.limiter.release()
                raise
            else:
                await slot.limiter.release(successful=True)
                result.concurrency_window = slot.limiter.window
                return result
            if attempt == 4:
                break
            await asyncio.sleep(min(20.0, 0.75 * (2**attempt)))
        raise RuntimeError(f"gateway call failed after retries: {last_error}") from last_error
