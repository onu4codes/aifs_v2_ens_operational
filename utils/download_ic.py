#!/usr/bin/env python
"""Download AIFS initial conditions (IFS oper + wave, step 0) from ECMWF open data.

For a cycle YYYYMMDDT00 this fetches four files into IC_data/YYYYMMDD/:
    <YYYYMMDD>000000-0h-oper-fc.grib2   and  -wave-fc.grib2   (t0)
    <YYYYMMDD-1>180000-0h-oper-fc.grib2 and  -wave-fc.grib2   (t-6h)

Usage:
    python utils/download_ic.py                  # latest available date, 00 UTC
    python utils/download_ic.py --date 20260920  # a specific date, 00 UTC

On success the cycle it downloaded (or found) is printed to stdout as YYYYMMDDT00, and
nothing else is ever printed, so a wrapper can capture it:
    CYCLE=$(python utils/download_ic.py) || exit $?
All other messages go to logs/AIFS.log. Check the result with the exit code ($?).

Exit codes:
    0  ICs available (already present or downloaded)
    1  download / network failure
    2  bad arguments
    3  data not available on the server
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from ecmwf.opendata import Client as OpendataClient

########################################################################################################
# Modify this part according to your needs:
REPO_ROOT = Path(__file__).resolve().parents[1]                  # aifs_v2_ens_operational/
DATA_ROOT = Path(os.environ.get("AIFS_DATA_ROOT", REPO_ROOT))    # override to keep data elsewhere
log_dir = DATA_ROOT / "logs"
IC_dir = DATA_ROOT / "IC_data"

# Open-data mirrors, tried in this order for every file. The ECMWF portal is limited to a
# fixed number of simultaneous connections for all users and often answers 429 at busy
# times; the cloud replicas have the same files and path layout. (Azure is not included:
# it needs a SAS token.) The AWS replica also keeps a longer archive than the portal.
SOURCES = ["aws", "google", "ecmwf"]
MIRRORS = {
    "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
    "google": "https://storage.googleapis.com/ecmwf-open-data",
    "ecmwf": "https://data.ecmwf.int/forecasts",
}
MODELS = {"oper": "oper-fc", "wave": "wave-fc"}   # URL folder -> filename suffix

TIMEOUT = (30, 300)        # (connect, read) seconds
RETRIES = 3                # per mirror, for network / server / 429 errors (404 moves on at once)
RETRY_WAIT = 60            # seconds; grows with each attempt, or follows the server's Retry-After
########################################################################################################

EXIT_OK, EXIT_DOWNLOAD_FAILED, EXIT_BAD_ARGS, EXIT_NOT_AVAILABLE = 0, 1, 2, 3

os.makedirs(log_dir, exist_ok=True)
os.makedirs(IC_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "AIFS.log")],
)
logger = logging.getLogger("download_ic")


class DataNotAvailable(Exception):
    """The server has no such file (HTTP 404)."""


class DownloadFailed(Exception):
    """Network or server error that persisted after retries on every mirror."""


def parse_date(value):
    try:
        return datetime.strptime(value, "%Y%m%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date '{value}', expected YYYYMMDD (e.g. 20260920)")


def check_latest_ic_date():
    """Latest cycle, asking each mirror in turn."""
    errors = []
    for source in SOURCES:
        try:
            latest = OpendataClient(source).latest()
            logger.info(f"Latest date from {source}: {latest}")
            return latest
        except Exception as exc:
            logger.warning(f"Could not get latest date from {source}: {exc}")
            errors.append(f"{source}: {exc}")
    raise DownloadFailed("could not determine the latest date from any mirror: "
                         + "; ".join(errors))


def expected_files(cycle):
    """(relative_path, local_path) for the four IC files of a 00 UTC cycle."""
    IC_loc = IC_dir / cycle.strftime("%Y%m%d")
    files = []
    for d in (cycle - timedelta(hours=6), cycle):
        for folder, suffix in MODELS.items():
            filename = f"{d.strftime('%Y%m%d%H%M%S')}-0h-{suffix}.grib2"
            rel = f"{d.strftime('%Y%m%d')}/{d.strftime('%Hz')}/ifs/0p25/{folder}/{filename}"
            files.append((rel, IC_loc / filename))
    return files


def _wait_time(exc, attempt):
    """Honour the server's Retry-After header if present, otherwise back off linearly."""
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.isdigit():
            return min(int(retry_after), 600)
    return RETRY_WAIT * attempt


def _with_retries(action, description):
    """Run action(); retry network/5xx/429 errors, but let DataNotAvailable through at once."""
    for attempt in range(1, RETRIES + 1):
        try:
            return action()
        except DataNotAvailable:
            raise
        except (requests.RequestException, OSError) as exc:
            if attempt == RETRIES:
                raise DownloadFailed(f"{description}: {exc}") from exc
            wait = _wait_time(exc, attempt)
            logger.warning(f"{description}: attempt {attempt}/{RETRIES} failed ({exc}); "
                           f"retrying in {wait}s")
            time.sleep(wait)


def find_source(rel):
    """Return the first mirror that has this file.

    Raises DataNotAvailable if every reachable mirror answers 404, and DownloadFailed if
    no mirror could be reached at all.
    """
    name = rel.rsplit("/", 1)[-1]
    failures = []
    for source in SOURCES:
        url = f"{MIRRORS[source]}/{rel}"
        try:
            probe(url, f"checking {name} on {source}")
            return source
        except DataNotAvailable:
            logger.info(f"{name} not found on {source}")
        except DownloadFailed as exc:
            logger.warning(f"{source} unreachable for {name}; trying next mirror")
            failures.append(str(exc))
    if len(failures) == len(SOURCES):
        raise DownloadFailed("; ".join(failures))
    raise DataNotAvailable(name)


def _raise_for_status(response, url):
    if response.status_code == 404:
        raise DataNotAvailable(url)
    response.raise_for_status()


def probe(url, description):
    """Check a file exists on the server without downloading it."""
    def action():
        r = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
        if r.status_code in (403, 405):          # server doesn't allow HEAD: fall back to GET
            with requests.get(url, stream=True, timeout=TIMEOUT) as g:
                _raise_for_status(g, url)
            return
        _raise_for_status(r, url)
    _with_retries(action, description)


def download(url, file_path, source):
    """Download to <file>.part and rename when complete, so a partial file is never kept."""
    part = file_path.with_name(file_path.name + ".part")

    def action():
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as r:
                _raise_for_status(r, url)
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            part.replace(file_path)
        finally:
            part.unlink(missing_ok=True)

    _with_retries(action, f"downloading {file_path.name} from {source}")
    size_mb = file_path.stat().st_size / 1e6
    logger.info(f"Successfully downloaded {file_path.name} from {source} ({size_mb:.0f} MB)")


def download_any(rel, file_path, first_source):
    """Download from first_source; if it keeps failing, fall back to the other mirrors."""
    order = [first_source] + [s for s in SOURCES if s != first_source]
    failures = []
    for source in order:
        url = f"{MIRRORS[source]}/{rel}"
        logger.info(f"Downloading {url}")
        try:
            download(url, file_path, source)
            return
        except DataNotAvailable:
            logger.info(f"{file_path.name} not found on {source}; trying next mirror")
            failures.append(f"{source}: not found")
        except DownloadFailed as exc:
            logger.warning(f"{source} failed for {file_path.name}; trying next mirror")
            failures.append(str(exc))
    raise DownloadFailed(f"{file_path.name} failed on every mirror: " + " | ".join(failures))


def unavailable_reason(cycle, from_latest):
    today = datetime.now(timezone.utc).date()
    if from_latest or cycle.date() >= today:
        return "it has probably not been published yet; try again later"
    return ("none of the mirrors has it. The ECMWF portal keeps only recent days; the "
            "cloud replicas keep longer but not forever, and file layouts changed over time.")


class LoggingArgumentParser(argparse.ArgumentParser):
    """Send argument errors to the log instead of the terminal."""

    def error(self, message):
        logger.error(f"Invalid arguments: {message}")
        sys.exit(EXIT_BAD_ARGS)


def main():
    parser = LoggingArgumentParser(description="Download AIFS initial conditions (00 UTC).")
    parser.add_argument("--date", type=parse_date,
                        help="date to download, YYYYMMDD (default: latest available)")
    args = parser.parse_args()

    logger.info("Starting the download process...")
    try:
        # 1. Resolve the target cycle
        if args.date:
            cycle = args.date
            if cycle.date() > datetime.now(timezone.utc).date():
                logger.error(f"{cycle:%Y%m%d} is in the future; no ICs exist for it yet.")
                sys.exit(EXIT_BAD_ARGS)
            logger.info(f"Requested date: {cycle:%Y%m%d}")
        else:
            latest = check_latest_ic_date()
            cycle = latest.replace(hour=0, minute=0, second=0, microsecond=0)
            logger.info(f"Latest IC date found: {latest}; using {cycle:%Y%m%dT%H}")

        cycle_str = cycle.strftime("%Y%m%dT%H")

        # 2. Already downloaded?
        files = expected_files(cycle)
        missing = [(url, path) for url, path in files if not path.exists()]
        if not missing:
            logger.info(f"ICs for {cycle_str} already available in {files[0][1].parent}. "
                        f"Skipping download.")
            print(cycle_str)
            sys.exit(EXIT_OK)

        logger.info(f"{len(missing)} of {len(files)} IC files missing for {cycle_str}")

        # 3. Find a mirror for every missing file before downloading anything
        plan, not_on_server = [], []
        for rel, path in missing:
            try:
                plan.append((rel, path, find_source(rel)))
            except DataNotAvailable:
                not_on_server.append(path.name)
        if not_on_server:
            logger.error(f"ICs for {cycle_str} are not available on the server "
                         f"(missing: {', '.join(not_on_server)}): "
                         f"{unavailable_reason(cycle, from_latest=not args.date)}")
            sys.exit(EXIT_NOT_AVAILABLE)

        # 4. Download
        files[0][1].parent.mkdir(parents=True, exist_ok=True)
        for rel, path, source in plan:
            download_any(rel, path, source)

        logger.info(f"ICs for {cycle_str} downloaded successfully.")
        print(cycle_str)
        sys.exit(EXIT_OK)

    except DataNotAvailable as e:
        logger.error(f"File disappeared from the server during download: {e}")
        sys.exit(EXIT_NOT_AVAILABLE)
    except DownloadFailed as e:
        logger.error(f"Download failed after {RETRIES} attempts per mirror: {e}")
        sys.exit(EXIT_DOWNLOAD_FAILED)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(EXIT_DOWNLOAD_FAILED)


if __name__ == "__main__":
    main()