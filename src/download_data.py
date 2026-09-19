"""Download the Milan dataset and shrink each day into a parquet file.

The files sit behind a Dataverse guestbook, so a plain download returns an error.
Posting contact details gives back a temporary link that does work.

Each raw day is about 320 MB of text and there are 62 of them, which is 20.8 GB. That
is too much to keep around, so every file is reduced to roughly 27 MB of parquet as
soon as it lands and the raw text is then zipped up.

Usage:
    DATAVERSE_EMAIL=you@example.com python src/download_data.py
"""

import argparse
import gzip
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ACTIVITY_COLUMNS,
    DAILY_DIRECTORY,
    LOG_DIRECTORY,
    RAW_COLUMNS,
    RAW_DIRECTORY,
)

DOI = "doi:10.7910/DVN/EGZHFV"
HOST = "https://dataverse.harvard.edu"
HEADERS = {"User-Agent": "milan-traffic-forecasting/1.0"}
STATE_FILE = LOG_DIRECTORY / "download_state.json"
MAX_TRIES = 10


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def get_json(url: str, body: dict | None = None) -> dict:
    """GET, or POST when a body is given. Retries a few times on network errors."""
    data = json.dumps(body).encode() if body else None
    headers = {**HEADERS, "Content-Type": "application/json"} if body else HEADERS

    for attempt in range(MAX_TRIES):
        try:
            request = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except Exception as error:
            if attempt == MAX_TRIES - 1:
                raise
            log(f"  {type(error).__name__}, retrying")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def list_files() -> list[dict[str, Any]]:
    """Ask Dataverse what is in the dataset."""
    payload = get_json(f"{HOST}/api/datasets/:persistentId/?persistentId={DOI}")
    files = [
        {
            "id": f["dataFile"]["id"],
            "name": f["dataFile"]["filename"],
            "size": f["dataFile"]["filesize"],
        }
        for f in payload["data"]["latestVersion"]["files"]
    ]
    return sorted(files, key=lambda f: f["name"])


def get_download_link(file_id: int, contact: dict[str, str]) -> str:
    """Answer the guestbook and get a temporary download link back."""
    payload = get_json(
        f"{HOST}/api/access/datafile/{file_id}", {"guestbookResponse": contact}
    )
    return payload["data"]["signedUrl"]


def size_of(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def download(
    file_id: int, target: Path, expected_size: int, contact: dict[str, str]
) -> None:
    """Download a file, picking up where it left off if it was interrupted."""
    partial = target.with_suffix(".part")

    for attempt in range(MAX_TRIES):
        have = size_of(partial)
        if have == expected_size:
            break
        if have > expected_size:  # a bad earlier attempt, start over
            partial.unlink()
            have = 0

        try:
            link = get_download_link(file_id, contact)
            request = urllib.request.Request(
                link, headers={**HEADERS, "Range": f"bytes={have}-"}
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                # If the server ignored our Range header it is sending the whole file,
                # so overwrite instead of appending or we would corrupt the file.
                mode = "ab" if have and response.status == 206 else "wb"
                with open(partial, mode) as handle:
                    shutil.copyfileobj(response, handle, 1 << 20)
        except Exception as error:
            log(f"  {target.name}: {type(error).__name__}, will resume")
            time.sleep(5)

    if size_of(partial) != expected_size:
        raise RuntimeError(f"{target.name} is incomplete after {MAX_TRIES} tries")
    partial.rename(target)


def reduce_day(raw_path: Path, out_path: Path) -> int:
    """Sum activity over country codes, so one row per area per 10 minutes.

    Read in chunks and with small dtypes, otherwise one day needs about 600 MB.
    """
    dtypes: dict[str, str] = {
        "square_id": "int16",
        "time_ms": "int64",
        "country_code": "int16",
    }
    dtypes.update({column: "float32" for column in ACTIVITY_COLUMNS})

    parts = []
    for chunk in pd.read_csv(
        raw_path, sep="\t", names=RAW_COLUMNS, dtype=dtypes, chunksize=2_000_000
    ):
        chunk = chunk.drop(columns="country_code")
        parts.append(chunk.groupby(["square_id", "time_ms"], sort=False).sum())

    # Groups can straddle two chunks, so add the partial sums together at the end.
    day = pd.concat(parts).groupby(level=[0, 1], sort=False).sum().reset_index()
    day["timestamp"] = pd.to_datetime(day.pop("time_ms"), unit="ms", utc=True)
    day.sort_values(["square_id", "timestamp"]).to_parquet(out_path, index=False)
    return len(day)


def zip_raw(raw_path: Path) -> None:
    """Keep the original text, but at about a quarter of the size."""
    with open(raw_path, "rb") as source, gzip.open(f"{raw_path}.gz", "wb") as target:
        shutil.copyfileobj(source, target, 8 << 20)
    raw_path.unlink()


def handle_file(
    entry: dict[str, Any], contact: dict[str, str], keep_raw: bool
) -> dict[str, Any]:
    """Download one day, reduce it, tidy up the raw text."""
    date = entry["name"].replace("sms-call-internet-mi-", "").replace(".txt", "")
    out_path = DAILY_DIRECTORY / f"{date}.parquet"
    raw_path = RAW_DIRECTORY / entry["name"]

    log(f"{date}: downloading {entry['size'] / 1e6:.0f} MB")
    start = time.time()
    download(entry["id"], raw_path, entry["size"], contact)

    try:
        rows = reduce_day(raw_path, out_path)
    except Exception:
        raw_path.unlink(missing_ok=True)  # do not reuse a broken file next run
        out_path.unlink(missing_ok=True)
        raise

    if not keep_raw:
        zip_raw(raw_path)

    log(
        f"{date}: {rows:,} rows, {out_path.stat().st_size / 1e6:.1f} MB parquet, {time.time() - start:.0f}s"
    )
    return {
        "date": date,
        "rows": rows,
        "raw_bytes": entry["size"],
        "parquet_bytes": out_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads")
    parser.add_argument("--limit", type=int, help="only do the first N days")
    parser.add_argument(
        "--keep-raw", action="store_true", help="do not zip the raw text"
    )
    args = parser.parse_args()

    contact = {
        "name": os.environ.get("DATAVERSE_NAME", "Anonymous"),
        "email": os.environ.get("DATAVERSE_EMAIL", ""),
        "institution": os.environ.get("DATAVERSE_INSTITUTION", ""),
        "position": os.environ.get("DATAVERSE_POSITION", "Student"),
    }
    if not contact["email"]:
        sys.exit("set DATAVERSE_EMAIL, the guestbook needs an email address")

    state: dict[str, Any] = (
        json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    )
    files = list_files()[: args.limit]
    todo = [f for f in files if f["name"] not in state]
    log(f"{len(files)} files, {len(todo)} still to do")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(
            lambda f: (f["name"], handle_file(f, contact, args.keep_raw)), todo
        )
        for name, record in results:
            state[name] = record
            STATE_FILE.write_text(json.dumps(state, indent=2))

    log("done")


if __name__ == "__main__":
    main()
