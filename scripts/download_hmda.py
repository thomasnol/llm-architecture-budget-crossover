from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import httpx

URL = (
    "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
    "?states=DC,ND,VT,WY&years=2024&actions_taken=1,3"
)
EXPECTED_SHA256 = "ed1f933f5b3487310c8364aebba8cb8b82d3f9870ff744648899e62baceaf4f5"
EXPECTED_HEADER_PREFIX = "activity_year,lei,derived_msa-md,state_code,"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    with httpx.stream("GET", URL, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_bytes():
                stream.write(chunk)
    with temporary.open("r", encoding="utf-8") as stream:
        header = stream.readline()
    if not header.startswith(EXPECTED_HEADER_PREFIX):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("download did not contain the expected HMDA CSV header")
    observed = sha256_file(temporary)
    if observed != EXPECTED_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "frozen HMDA checksum changed; do not silently accept a new source. "
            f"Expected {EXPECTED_SHA256}, observed {observed}."
        )
    shutil.move(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the pinned official 2024 HMDA experiment cohort."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/hmda_2024_dc_nd_vt_wy.csv"),
    )
    args = parser.parse_args()
    if args.destination.exists():
        observed = sha256_file(args.destination)
        if observed != EXPECTED_SHA256:
            raise RuntimeError(f"{args.destination} exists with unexpected checksum {observed}")
        print(f"Verified existing {args.destination} ({observed})")
        return
    download(args.destination)
    print(f"Downloaded {args.destination} ({EXPECTED_SHA256})")


if __name__ == "__main__":
    main()
