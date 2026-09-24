#!/usr/bin/env python
"""Run the AIFS ENS v2 ensemble for one 00 UTC cycle.

Reads the input state written by preprocess_ic.py:
    IC_data/YYYYMMDD/input_state_YYYYMMDDT00_v2.pkl
runs N_MEMBERS members on all visible GPUs (one worker per GPU), regrids every step
from N320 to global 0.25 deg and writes the raw (unclipped) forecast to
    Outputs/raw/init_YYYYMMDDT00.zarr
with dims (time, number, prediction_timedelta, lat, lon).

Usage:
    python utils/run_aifs_ens.py --date 20260923T00

Nothing is printed to the terminal; everything (including library output) goes to
logs/AIFS.log.
Exit codes:
    0  forecast available (already present or created)
    1  run failed
    2  bad arguments
    3  input state missing (run preprocess_ic.py first)
"""

import argparse
import gc
import logging
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

###########################################################################################################
# Modify this part according to your needs:
REPO_ROOT = Path(__file__).resolve().parents[1]                  # aifs_v2_ens_operational/
DATA_ROOT = Path(os.environ.get("AIFS_DATA_ROOT", REPO_ROOT))
log_dir = DATA_ROOT / "logs"
IC_dir = DATA_ROOT / "IC_data"
OUTPUT_DIR = DATA_ROOT / "Outputs" / "raw"

CHECKPOINT_PATH = Path(os.environ.get(
    "AIFS_CHECKPOINT", REPO_ROOT / "weights" / "aifs-ens-crps-2.0.ckpt"))
# N320 -> 0.25 deg regular lat/lon interpolation matrix (MIR linear)
REGRID_MATRIX = Path(os.environ.get(
    "AIFS_REGRID_MATRIX_N320_LATLON",
    REPO_ROOT / "EKR" / "mir_16_linear"
    / "7f0be51c7c1f522592c7639e0d3f95bcbff8a044292aa281c1e73b842736d9bf.npz"))

N_MEMBERS = 51
LEAD_TIME_HOURS = 24 * 7            # must be a multiple of 6
SAVE_FIELDS = ["tp"]                # model output variables to store
###########################################################################################################

EXIT_OK, EXIT_FAILED, EXIT_BAD_ARGS, EXIT_MISSING_INPUT = 0, 1, 2, 3

N320_POINTS = 542080
LATITUDES = np.linspace(90, -90, 721)
LONGITUDES = np.linspace(0, 359.75, 1440)
LOG_FILE = log_dir / "AIFS.log"

os.makedirs(log_dir, exist_ok=True)


def setup_logging(process_name):
    """Log to logs/AIFS.log only, and send anything libraries print (stdout/stderr,
    including C-level CUDA messages and progress bars) to the same file."""
    fd = os.open(LOG_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - {process_name} - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_FILE)],
        force=True,
    )
    logging.captureWarnings(True)
    return logging.getLogger("run_aifs_ens")


class MissingInput(Exception):
    """The preprocessed input state is not on disk."""


class LoggingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        logging.getLogger("run_aifs_ens").error(f"Invalid arguments: {message}")
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


def member_seed(date, ens_number):
    """Reproducible, but different for every cycle and member."""
    return int(date.strftime("%Y%m%d")) * 100 + int(ens_number)


def load_input_state(date):
    date_f = date.strftime("%Y%m%dT%H")
    ic_path = IC_dir / date.strftime("%Y%m%d") / f"input_state_{date_f}_v2.pkl"
    if not ic_path.exists():
        raise MissingInput(str(ic_path))
    with open(ic_path, "rb") as f:
        state = pickle.load(f)
    if pd.Timestamp(state["date"]) != pd.Timestamp(date):
        raise ValueError(f"{ic_path} is for {state['date']}, expected {date}")
    shapes = {v.shape for v in state["fields"].values()}
    if shapes != {(2, N320_POINTS)}:
        raise ValueError(f"{ic_path}: unexpected field shapes {shapes}, want (2, {N320_POINTS})")
    logging.getLogger("run_aifs_ens").info(
        f"Loaded input state {ic_path} ({len(state['fields'])} fields)")
    return state


# ── worker: runs members on one GPU ──────────────────────────────────────────
def regrid_step(fields, matrix):
    data_vars = {}
    for name, values in fields.items():
        grid = (matrix @ np.asarray(values).reshape(-1)).reshape(721, 1440)
        data_vars[name] = (["lat", "lon"], grid.astype(np.float32))
    return data_vars


def run_member(runner, matrix, input_state, ens_number, date, logger):
    import torch
    import xarray as xr

    seed = member_seed(date, ens_number)
    torch.manual_seed(seed)
    logger.info(f"Member {ens_number}: seed {seed}")

    steps, datasets = [], []
    for state in runner.run(input_state=input_state, lead_time=LEAD_TIME_HOURS):
        step = int((pd.Timestamp(state["date"]) - pd.Timestamp(date)) / pd.Timedelta(hours=1))
        if step <= 0:
            continue  # the initial state is not a forecast
        missing = [v for v in SAVE_FIELDS if v not in state["fields"]]
        if missing:
            raise KeyError(f"Model output has no {missing}; "
                           f"available: {sorted(state['fields'])}")
        fields = {v: state["fields"][v] for v in SAVE_FIELDS}
        datasets.append(xr.Dataset(regrid_step(fields, matrix),
                                   coords={"lat": LATITUDES, "lon": LONGITUDES}))
        steps.append(step)
        logger.info(f"Member {ens_number}: step +{step}h done")

    expected = list(range(6, LEAD_TIME_HOURS + 1, 6))
    if steps != expected:
        raise RuntimeError(f"Member {ens_number}: got steps {steps}, expected {expected}")

    ds = xr.concat(datasets, dim="prediction_timedelta")
    ds["prediction_timedelta"] = pd.to_timedelta(steps, unit="h")
    ds = ds.expand_dims(number=[int(ens_number)]).expand_dims(time=[pd.Timestamp(date)])
    ds = ds.transpose("time", "number", "prediction_timedelta", "lat", "lon")
    ds.attrs.update(
        model="AIFS ENS v2", checkpoint=CHECKPOINT_PATH.name,
        initial_conditions="IFS HRES analysis (ECMWF open data), same for all members",
        source="Contains modified ECMWF data and model weights (CC BY 4.0)",
    )
    return ds.chunk({"time": 1, "number": 1, "prediction_timedelta": -1, "lat": -1, "lon": -1})


def worker_loop(worker_id, gpu_id, task_queue, input_state, date, result_queue, stop_event):
    logger = setup_logging(f"worker{worker_id}")
    try:
        import torch
        from anemoi.inference.runners.simple import SimpleRunner
        from scipy.sparse import load_npz

        torch.cuda.set_device(gpu_id)
        matrix = load_npz(REGRID_MATRIX)
        runner = SimpleRunner(str(CHECKPOINT_PATH), device=f"cuda:{gpu_id}")  # loaded once
        logger.info(f"Worker {worker_id}: model loaded on GPU {gpu_id} "
                    f"({torch.cuda.get_device_name(gpu_id)})")

        while not stop_event.is_set():
            try:
                ens_number = task_queue.get(timeout=5)
            except queue.Empty:
                continue
            if ens_number is None or stop_event.is_set():
                break

            logger.info(f"Worker {worker_id} starting member {ens_number}")
            ds = run_member(runner, matrix, input_state, ens_number, date, logger)
            while not stop_event.is_set():
                try:
                    result_queue.put(("member", ens_number, ds), timeout=5)
                    logger.info(f"Worker {worker_id} completed member {ens_number}")
                    break
                except queue.Full:
                    continue
            del ds
            gc.collect()
            torch.cuda.empty_cache()

        result_queue.put(("done", worker_id))
    except BaseException:
        logger.error(f"Worker {worker_id} failed:\n{traceback.format_exc()}")
        stop_event.set()
        result_queue.put(("error", worker_id, traceback.format_exc()))


# ── writer: appends members to the Zarr store in member order ────────────────
def writer_loop(result_queue, status_queue, stop_event, filename, final_filename,
                n_members, n_workers):
    logger = setup_logging("writer")
    pending, next_member, done_workers = {}, 0, set()
    try:
        while next_member < n_members:
            try:
                message = result_queue.get(timeout=5)
            except queue.Empty:
                if stop_event.is_set():
                    raise RuntimeError("Writer stopped before all members were received.")
                continue

            kind = message[0]
            if kind == "member":
                _, ens_number, ds = message
                pending[ens_number] = ds
                while next_member in pending:
                    ds = pending.pop(next_member)
                    if next_member == 0:
                        ds.to_zarr(filename, zarr_format=2, mode="w")
                    else:
                        ds.to_zarr(filename, zarr_format=2, mode="a", append_dim="number")
                    logger.info(f"Wrote member {next_member} to {filename.name}")
                    del ds
                    gc.collect()
                    next_member += 1
            elif kind == "done":
                done_workers.add(message[1])
                if len(done_workers) == n_workers and next_member < n_members:
                    missing = [m for m in range(next_member, n_members) if m not in pending]
                    raise RuntimeError(f"All workers finished early. Missing members: {missing}")
            elif kind == "error":
                stop_event.set()
                raise RuntimeError(f"Worker {message[1]} failed:\n{message[2]}")
            else:
                raise RuntimeError(f"Unknown writer queue message: {kind}")

        filename.rename(final_filename)
        status_queue.put(("done", str(final_filename)))
    except BaseException:
        logger.error(f"Writer failed:\n{traceback.format_exc()}")
        stop_event.set()
        status_queue.put(("error", "writer", traceback.format_exc()))


# ── main process ─────────────────────────────────────────────────────────────
def run_model(date, logger):
    import torch

    date_f = date.strftime("%Y%m%dT%H")
    filename = OUTPUT_DIR / f"init_{date_f}_partial.zarr"
    final_filename = OUTPUT_DIR / f"init_{date_f}.zarr"

    for path, what in ((CHECKPOINT_PATH, "Checkpoint"), (REGRID_MATRIX, "Regrid matrix")):
        if not path.exists():
            raise FileNotFoundError(f"{what} not found: {path}")
    if LEAD_TIME_HOURS <= 0 or LEAD_TIME_HOURS % 6:
        raise ValueError("LEAD_TIME_HOURS must be a positive multiple of 6")

    ngpus = torch.cuda.device_count()
    if ngpus < 1:
        raise RuntimeError("No CUDA GPUs detected.")
    logger.info(f"Detected {ngpus} CUDA GPUs")

    input_state = load_input_state(date)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if filename.exists():
        logger.warning(f"Removing leftover partial output {filename}")
        shutil.rmtree(filename)

    n_workers = min(ngpus, N_MEMBERS)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue(maxsize=max(2, n_workers))
    status_queue = ctx.Queue()
    stop_event = ctx.Event()

    for ens_number in range(N_MEMBERS):
        task_queue.put(ens_number)
    for _ in range(n_workers):
        task_queue.put(None)

    writer = ctx.Process(
        target=writer_loop, name="aifs-ens-writer",
        args=(result_queue, status_queue, stop_event, filename, final_filename,
              N_MEMBERS, n_workers))
    writer.start()

    workers = []
    for worker_id in range(n_workers):
        worker = ctx.Process(
            target=worker_loop, name=f"aifs-ens-worker-{worker_id}",
            args=(worker_id, worker_id, task_queue, input_state, date, result_queue, stop_event))
        worker.start()
        workers.append(worker)
    logger.info(f"Started {n_workers} workers for {N_MEMBERS} members, "
                f"lead time {LEAD_TIME_HOURS} h, fields {SAVE_FIELDS}")

    try:
        while True:
            try:
                status = status_queue.get(timeout=5)
            except queue.Empty:
                failed = [w for w in workers if w.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError("Worker process died: " + ", ".join(
                        f"{w.name} exit {w.exitcode}" for w in failed))
                if writer.exitcode is not None:
                    raise RuntimeError(f"Writer process exited with code {writer.exitcode}")
                continue
            if status[0] == "done":
                break
            raise RuntimeError(f"{status[1]} failed:\n{status[2]}")
    except BaseException:
        stop_event.set()
        for p in workers + [writer]:
            if p.is_alive():
                p.terminate()
        raise
    finally:
        for p in workers + [writer]:
            p.join()

    return final_filename


def main():
    logger = setup_logging("main")
    parser = LoggingArgumentParser(description="Run AIFS ENS v2 inference for one 00 UTC cycle.")
    parser.add_argument("--date", type=parse_date, required=True,
                        help="initialization date in YYYYMMDDT00 format, e.g. 20260923T00")
    args = parser.parse_args()
    date = args.date
    date_f = date.strftime("%Y%m%dT%H")

    logger.info(f"Starting AIFS ENS v2 inference for {date_f}")
    final_filename = OUTPUT_DIR / f"init_{date_f}.zarr"
    if final_filename.exists():
        logger.info(f"Forecast already exists: {final_filename}. Skipping model run.")
        sys.exit(EXIT_OK)

    try:
        result = run_model(date, logger)
        logger.info(f"Model run complete. Output saved to {result}")
        sys.exit(EXIT_OK)
    except MissingInput as e:
        logger.error(f"Input state missing (run preprocess_ic.py --date {date_f} first): {e}")
        sys.exit(EXIT_MISSING_INPUT)
    except Exception as e:
        logger.exception(f"Inference failed for {date_f}: {e}")
        sys.exit(EXIT_FAILED)


if __name__ == "__main__":
    main()