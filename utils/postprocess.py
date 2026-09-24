#!/usr/bin/env python
"""Clip the raw AIFS ENS v2 forecast to the region of interest and save it as NetCDF.

Reads   Outputs/raw/init_YYYYMMDDT00.zarr            (written by run_aifs_ens.py)
Writes  Outputs/postprocessed/init_YYYYMMDDT00.nc

Precipitation (tp) is converted from m to mm (6-hourly accumulation per step).

Usage:
    python utils/postprocess.py --date 20260923T00

Nothing is printed to the terminal; all messages go to logs/AIFS.log.
Exit codes:
    0  postprocessed file available (already present or created)
    1  postprocessing failed
    2  bad arguments
    3  raw forecast missing (run run_aifs_ens.py first)
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

###########################################################################################################
# Modify this part according to your needs:
REPO_ROOT = Path(__file__).resolve().parents[1]                  # aifs_v2_ens_operational/
DATA_ROOT = Path(os.environ.get("AIFS_DATA_ROOT", REPO_ROOT))
log_dir = DATA_ROOT / "logs"
RAW_DIR = DATA_ROOT / "Outputs" / "raw"
POST_DIR = DATA_ROOT / "Outputs" / "postprocessed"

# Region of interest. Longitudes are 0-360, as in the raw output.
LAT_MIN, LAT_MAX = 6.5, 38.5
LON_MIN, LON_MAX = 66.5, 100.0

COMPRESSION_LEVEL = 4               # zlib level for the NetCDF variables (1-9)

# Unit conversions applied after clipping: variable -> (factor, new units, long name).
# AIFS outputs tp in metres, accumulated over the 6 h ending at each step.
UNIT_CONVERSIONS = {
    "tp": (1000.0, "mm", "total precipitation accumulated over the preceding 6 h"),
}
###########################################################################################################

EXIT_OK, EXIT_FAILED, EXIT_BAD_ARGS, EXIT_MISSING_INPUT = 0, 1, 2, 3

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "AIFS.log")],
)
logging.captureWarnings(True)
logger = logging.getLogger("postprocess")


class MissingInput(Exception):
    """The raw forecast is not on disk."""


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


def open_raw(date_f):
    raw_file = RAW_DIR / f"init_{date_f}.zarr"
    if not raw_file.exists():
        partial = RAW_DIR / f"init_{date_f}_partial.zarr"
        hint = " (only a partial store exists: the model run did not finish)" if partial.exists() else ""
        raise MissingInput(f"{raw_file}{hint}")
    ds = xr.open_zarr(raw_file)
    logger.info(f"Opened {raw_file}: members {ds.sizes.get('number')}, "
                f"steps {ds.sizes.get('prediction_timedelta')}, variables {list(ds.data_vars)}")
    return ds, raw_file


def clip(ds):
    """Cut the region out of the global grid (works for ascending or descending latitude)."""
    lat_slice = (slice(LAT_MAX, LAT_MIN) if ds.lat.values[0] > ds.lat.values[-1]
                 else slice(LAT_MIN, LAT_MAX))
    region = ds.sel(lat=lat_slice, lon=slice(LON_MIN, LON_MAX))
    if region.sizes["lat"] == 0 or region.sizes["lon"] == 0:
        raise ValueError(f"Region {LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E selects no grid "
                         f"points (grid lat {float(ds.lat.min())}..{float(ds.lat.max())}, "
                         f"lon {float(ds.lon.min())}..{float(ds.lon.max())})")
    logger.info(f"Clipped to lat {float(region.lat.max())}..{float(region.lat.min())}, "
                f"lon {float(region.lon.min())}..{float(region.lon.max())} "
                f"({region.sizes['lat']} x {region.sizes['lon']} points)")
    return region


def convert_units(region):
    """Apply UNIT_CONVERSIONS to the variables that are present."""
    for name, (factor, units, long_name) in UNIT_CONVERSIONS.items():
        if name not in region:
            continue
        da = region[name].load()
        max_value = float(da.max())
        # 6 h of rain never exceeds 1 m, so a larger maximum means the data are already in mm.
        if max_value > 1.0:
            logger.warning(f"'{name}' max {max_value:.2f} looks already converted "
                           f"(> 1 m per 6 h); leaving values unchanged")
        else:
            da = da * factor
            logger.info(f"Converted '{name}' to {units} (x{factor:g}); "
                        f"max {float(da.max()):.2f} {units}")
        da.attrs.update(units=units, long_name=long_name)
        region[name] = da.astype("float32")
    return region


def save_netcdf(region, out_file, raw_file):
    region = region.load()
    region.attrs.update(
        region=f"lat {LAT_MIN} to {LAT_MAX}, lon {LON_MIN} to {LON_MAX} (0-360)",
        source_file=raw_file.name,
        history=f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC: clipped by postprocess.py",
    )
    region.lat.attrs.update(units="degrees_north", standard_name="latitude")
    region.lon.attrs.update(units="degrees_east", standard_name="longitude")

    encoding = {name: {"zlib": True, "complevel": COMPRESSION_LEVEL, "dtype": "float32"}
                for name in region.data_vars}
    for name in region.variables:          # drop zarr-specific chunk/compressor settings
        region[name].encoding = {}
    encoding["prediction_timedelta"] = {"units": "hours", "dtype": "int32"}

    tmp = out_file.with_name(out_file.name + ".part")
    try:
        region.to_netcdf(tmp, engine="netcdf4", encoding=encoding)
        tmp.replace(out_file)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info(f"Saved {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = LoggingArgumentParser(description="Clip the raw AIFS ENS v2 forecast to the region.")
    parser.add_argument("--date", type=parse_date, required=True,
                        help="initialization date in YYYYMMDDT00 format, e.g. 20260923T00")
    args = parser.parse_args()
    date_f = args.date.strftime("%Y%m%dT%H")

    logger.info(f"Starting postprocessing for {date_f}")
    out_file = POST_DIR / f"init_{date_f}.nc"
    if out_file.exists():
        logger.info(f"Postprocessed file already exists: {out_file}. Skipping.")
        sys.exit(EXIT_OK)

    try:
        ds, raw_file = open_raw(date_f)
        region = convert_units(clip(ds))
        POST_DIR.mkdir(parents=True, exist_ok=True)
        save_netcdf(region, out_file, raw_file)
        logger.info(f"Postprocessing for {date_f} complete.")
        sys.exit(EXIT_OK)
    except MissingInput as e:
        logger.error(f"Raw forecast missing (run run_aifs_ens.py --date {date_f} first): {e}")
        sys.exit(EXIT_MISSING_INPUT)
    except Exception as e:
        logger.exception(f"Postprocessing failed for {date_f}: {e}")
        sys.exit(EXIT_FAILED)


if __name__ == "__main__":
    main()