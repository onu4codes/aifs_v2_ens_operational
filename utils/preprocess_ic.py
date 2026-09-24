#!/usr/bin/env python
"""Build the AIFS ENS v2 input state from downloaded ICs.

Reads the four files written by download_ic.py from IC_data/YYYYMMDD/:
    <YYYYMMDD-1>180000-0h-oper-fc.grib2, -wave-fc.grib2   (t-6h)
    <YYYYMMDD>000000-0h-oper-fc.grib2,   -wave-fc.grib2   (t0)
regrids them from 0.25 deg to N320 and writes
    IC_data/YYYYMMDD/input_state_YYYYMMDDT00_v2.pkl

Usage:
    python utils/preprocess_ic.py --date 20260923T00

Nothing is printed to the terminal; all messages go to logs/AIFS.log.
Exit codes:
    0  input state available (already present or created)
    1  preprocessing failed
    2  bad arguments
    3  input GRIB files missing (run download_ic.py first)
"""

import argparse
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

###########################################################################################################
# Modify this part according to your needs:
REPO_ROOT = Path(__file__).resolve().parents[1]                  # aifs_v2_ens_operational/
DATA_ROOT = Path(os.environ.get("AIFS_DATA_ROOT", REPO_ROOT))
log_dir = DATA_ROOT / "logs"
IC_dir = DATA_ROOT / "IC_data"

# 0.25 deg regular lat/lon -> N320 interpolation matrix (MIR linear, from the earthkit-regrid cache)
REGRID_MATRIX = Path(os.environ.get(
    "AIFS_REGRID_MATRIX",
    REPO_ROOT / "EKR" / "mir_16_linear"
    / "9533e90f8433424400ab53c7fafc87ba1a04453093311c0b5bd0b35fedc1fb83.npz"))
# native N320 land-sea mask
LSM_GRIB_PATH = Path(os.environ.get("AIFS_LSM_GRIB", REPO_ROOT / "EKR" / "lsm.grib"))
###########################################################################################################

EXIT_OK, EXIT_FAILED, EXIT_BAD_ARGS, EXIT_MISSING_INPUT = 0, 1, 2, 3

OPEN_DATA_SHAPE = (721, 1440)
N320_POINTS = 542080

# ── v2 parameters ────────────────────────────────────────────────────────────
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor", "sd"]
PARAM_SOIL = ["vsw", "sot"]
PARAM_WAVE = ["wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530", "mwd", "cdww", "mwp", "swh"]
# w and q_50 are diagnostic (output-only) in v2, but the checkpoint still expects them in the
# input state (the run fails with "Missing variables" otherwise), as in ECMWF's v2 notebook.
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
SOIL_LEVELS = [1, 2]

SOIL_MAPPING = {"sot_1": "stl1", "sot_2": "stl2", "vsw_1": "swvl1", "vsw_2": "swvl2"}
REMOVE_FIELDS = ["q_10"]                # 10 hPa humidity is not used by the model
MASKED_FIELDS = ["sd", "swvl1", "swvl2"]
GRAVITY = 9.80665

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "AIFS.log")],
)
logger = logging.getLogger("preprocess_ic")


class MissingInput(Exception):
    """An input GRIB file is not on disk."""


class LoggingArgumentParser(argparse.ArgumentParser):
    """Send argument errors to the log instead of the terminal."""

    def error(self, message):
        logger.error(f"Invalid arguments: {message}")
        sys.exit(EXIT_BAD_ARGS)


def parse_date(value):
    try:
        date = datetime.strptime(value, "%Y%m%dT%H")
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid date '{value}', expected YYYYMMDDT00 (e.g. 20260923T00)")
    if date.hour != 0:
        raise argparse.ArgumentTypeError(f"only 00 UTC cycles are supported, got '{value}'")
    return date


# ── helpers ──────────────────────────────────────────────────────────────────
def load_regrid_matrix():
    from scipy.sparse import load_npz

    if not REGRID_MATRIX.exists():
        raise FileNotFoundError(f"Regrid matrix not found: {REGRID_MATRIX}")
    matrix = load_npz(REGRID_MATRIX)
    expected = (N320_POINTS, OPEN_DATA_SHAPE[0] * OPEN_DATA_SHAPE[1])
    if matrix.shape != expected:
        raise ValueError(f"Regrid matrix has shape {matrix.shape}, expected {expected}")
    logger.info(f"Loaded regrid matrix {REGRID_MATRIX.name}")
    return matrix


def to_n320(values, matrix):
    """Shift -180..180 -> 0..360 and interpolate 0.25 deg -> N320."""
    if values.shape != OPEN_DATA_SHAPE:
        raise ValueError(f"Unexpected field shape {values.shape}, expected {OPEN_DATA_SHAPE}")
    values = np.roll(values, -values.shape[1] // 2, axis=1)
    return matrix @ values.flatten()


def input_file(date, d, kind):
    return IC_dir / date.strftime("%Y%m%d") / f"{d.strftime('%Y%m%d%H%M%S')}-0h-{kind}-fc.grib2"


def check_inputs(date):
    missing = [input_file(date, d, kind)
               for d in (date - timedelta(hours=6), date) for kind in ("oper", "wave")]
    missing = [p for p in missing if not p.exists()]
    if missing:
        raise MissingInput(", ".join(str(p) for p in missing))


def extract(source, selection, with_level, label, d, fields, matrix):
    """Append every selected field (regridded) to fields[name]."""
    count = 0
    for f in source.sel(**selection):
        name = f"{f.metadata('param')}_{f.metadata('level')}" if with_level else f.metadata("param")
        fields.setdefault(name, []).append(to_n320(f.to_numpy(), matrix))
        count += 1
    logger.info(f"Extracted {count} {label} fields from {d.strftime('%Y%m%d%H%M%S')}")


def read_all_fields(date, matrix):
    """Read and regrid all v2 fields for t-6h and t0. Returns {name: array(2, N320_POINTS)}."""
    import earthkit.data as ekd

    dates = [date - timedelta(hours=6), date]
    fields = {}
    for d in dates:
        oper_path = input_file(date, d, "oper")
        wave_path = input_file(date, d, "wave")

        logger.info(f"Reading GRIB file: {oper_path}")
        oper = ekd.from_source("file", str(oper_path))
        extract(oper, dict(param=PARAM_SFC, levtype="sfc"), False, "surface", d, fields, matrix)
        extract(oper, dict(param=PARAM_SOIL, level=SOIL_LEVELS), True, "soil", d, fields, matrix)
        extract(oper, dict(param=PARAM_PL, level=LEVELS), True, "pressure-level", d, fields, matrix)

        logger.info(f"Reading GRIB file: {wave_path}")
        wave = ekd.from_source("file", str(wave_path))
        extract(wave, dict(param=PARAM_WAVE), False, "wave", d, fields, matrix)

    # every field must exist exactly once per input time
    expected = set(PARAM_SFC) | set(PARAM_WAVE)
    expected |= {f"{p}_{lev}" for p in PARAM_SOIL for lev in SOIL_LEVELS}
    expected |= {f"{p}_{lev}" for p in PARAM_PL for lev in LEVELS}
    missing = sorted(expected - set(fields))
    if missing:
        raise ValueError(f"Missing parameters: {missing}")
    wrong = {k: len(v) for k, v in fields.items() if len(v) != len(dates)}
    if wrong:
        raise ValueError(f"Fields not present exactly once for both t-6h and t0: {wrong}")

    return {k: np.stack(v) for k, v in fields.items()}


def apply_transforms(fields):
    # soil names
    for old, new in SOIL_MAPPING.items():
        fields[new] = fields.pop(old)
    logger.info("Renamed soil fields: " + ", ".join(f"{k}->{v}" for k, v in SOIL_MAPPING.items()))

    # mean wave direction -> sin/cos
    mwd = np.deg2rad(fields.pop("mwd"))
    fields["cos_mwd"] = np.cos(mwd)
    fields["sin_mwd"] = np.sin(mwd)
    logger.info("Transformed 'mwd' -> 'cos_mwd', 'sin_mwd'")

    # geopotential height -> geopotential
    for level in LEVELS:
        fields[f"z_{level}"] = fields.pop(f"gh_{level}") * GRAVITY
    logger.info("Transformed 'gh_*' -> 'z_*'")

    for var in REMOVE_FIELDS:
        if fields.pop(var, None) is not None:
            logger.info(f"Removed '{var}'")
    return fields


def apply_land_sea_mask(fields):
    import earthkit.data as ekd

    if not LSM_GRIB_PATH.exists():
        raise FileNotFoundError(f"LSM file not found: {LSM_GRIB_PATH}")
    lsm = ekd.from_source("file", str(LSM_GRIB_PATH))[0].to_numpy(flatten=True)
    if lsm.shape != (N320_POINTS,):
        raise ValueError(f"LSM has shape {lsm.shape}, expected ({N320_POINTS},)")
    sea = np.equal(lsm, 0)
    for name in MASKED_FIELDS:
        fields[name][:, sea] = np.nan
    logger.info(f"Land-sea mask applied to {MASKED_FIELDS} ({sea.sum()} sea points)")
    return fields


def save_state(state, output_file):
    """Write to a temporary file and rename, so a partial pickle is never left behind."""
    tmp = output_file.with_name(output_file.name + ".part")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(output_file)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info(f"v2 input state saved to: {output_file} "
                f"({output_file.stat().st_size / 1e9:.2f} GB)")


def preprocess_v2(date, output_file):
    logger.info("Starting v2 preprocessing...")
    matrix = load_regrid_matrix()
    fields = read_all_fields(date, matrix)
    fields = apply_transforms(fields)
    fields = apply_land_sea_mask(fields)
    logger.info(f"Total v2 fields: {len(fields)}")
    save_state(dict(date=date, fields=fields), output_file)


def main():
    parser = LoggingArgumentParser(description="Preprocess AIFS ENS v2 initial conditions.")
    parser.add_argument("--date", type=parse_date, required=True,
                        help="cycle in YYYYMMDDT00 format, e.g. 20260923T00")
    args = parser.parse_args()
    date = args.date
    cycle_str = date.strftime("%Y%m%dT%H")

    logger.info(f"Starting the preprocessing of initial conditions for {cycle_str}...")
    output_file = IC_dir / date.strftime("%Y%m%d") / f"input_state_{cycle_str}_v2.pkl"

    if output_file.exists():
        logger.info(f"v2 input state already exists: {output_file}. Skipping preprocessing.")
        sys.exit(EXIT_OK)

    try:
        check_inputs(date)
        preprocess_v2(date, output_file)
        logger.info(f"Input state for {cycle_str} preprocessed successfully.")
        sys.exit(EXIT_OK)
    except MissingInput as e:
        logger.error(f"Input files missing (run download_ic.py --date {date:%Y%m%d} first): {e}")
        sys.exit(EXIT_MISSING_INPUT)
    except Exception as e:
        logger.exception(f"Preprocessing failed for {cycle_str}: {e}")
        sys.exit(EXIT_FAILED)


if __name__ == "__main__":
    main()
    