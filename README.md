# AIFS ENS v2 operational pipeline

Runs ECMWF's **AIFS ENS v2** machine-learning ensemble forecast for a 00 UTC cycle, starting from
ECMWF open data, and produces a regional NetCDF file with the forecast variables you choose.

```
 download_ic.py  ──>  preprocess_ic.py  ──>  submit_aifs_v2_ens.sh  ──>  postprocess.py
   (CPU, internet)      (CPU)                 (GPU job: run_aifs_ens.py)    (CPU)
   IC files, 0.25°      model input state     global raw forecast           regional NetCDF
```

One command runs all of it: `./run_aifs_pipeline.sh`.

---

## Contents

1. [Repository layout](#repository-layout)
2. [Requirements](#requirements)
3. [Setting up the environment](#setting-up-the-environment)
4. [First-time setup on a new system](#first-time-setup-on-a-new-system)
5. [Running the pipeline](#running-the-pipeline)
6. [Configuration](#configuration)
7. [Outputs](#outputs)
8. [Logs and exit codes](#logs-and-exit-codes)
9. [Troubleshooting](#troubleshooting)
10. [Scientific notes and limitations](#scientific-notes-and-limitations)

---

## Repository layout

```
aifs_v2_ens_operational/
├── Readme.md
├── aifs_ens_v2.yml              conda environment definition
├── run_aifs_pipeline.sh         wrapper: runs the whole pipeline for one cycle (CPU / login node)
├── utils/
│   ├── download_ic.py           1. download initial conditions from ECMWF open data
│   ├── preprocess_ic.py         2. build the model input state (regrid to N320, transforms)
│   ├── submit_aifs_v2_ens.sh    3. Slurm job script for the GPU step
│   ├── run_aifs_ens.py             run by the job: the ensemble forecast
│   └── postprocess.py           4. clip to the region, convert units, write NetCDF
├── EKR/
│   ├── lsm.grib                 native N320 land-sea mask
│   └── mir_16_linear/           interpolation matrices
│       ├── 9533e90f…fc83.npz       0.25° lat/lon -> N320   (used by preprocess_ic.py)
│       └── 7f0be51c…d9bf.npz       N320 -> 0.25° lat/lon   (used by run_aifs_ens.py)
├── weights/
│   └── aifs-ens-crps-2.0.ckpt   model checkpoint
├── IC_data/YYYYMMDD/            downloaded GRIB files and input_state_YYYYMMDDT00_v2.pkl
├── Outputs/
│   ├── raw/                     init_YYYYMMDDT00.zarr   global forecast (0.25°)
│   └── postprocessed/           init_YYYYMMDDT00.nc     regional forecast
└── logs/                        AIFS.log and Slurm .out files
```

Code lives in the repository root and `utils/`. `EKR/` and `weights/` hold large static files,
and `IC_data/`, `Outputs/` and `logs/` hold data produced by the pipeline. Keep all five out of
version control (add them to `.gitignore`).

---

## Requirements

- **Linux** with **Slurm**.
- **NVIDIA GPU**, Ampere generation or newer (for example A100, L40S, H100). FlashAttention does
  not support older GPUs.
- **NVIDIA driver** supporting CUDA 12.6 or newer.
- **Conda** (Miniconda, Miniforge or Anaconda).
- **Internet access from the node that runs the wrapper** (usually the login node). Only the
  download step needs it; the GPU job runs offline.
- **Disk space** per cycle: about 1 GB for the model input state plus a few hundred MB of GRIB
  files in `IC_data/`, and the forecast output in `Outputs/`. The raw global forecast is by far
  the largest item; its size grows with the number of members, lead time and saved variables.

---

## Setting up the environment

The environment is defined in `aifs_ens_v2.yml`. Packages that determine how the model and its
inputs behave are pinned exactly (torch, anemoi, earthkit-regrid, ecmwf-opendata); the others
have version ranges so that each system gets versions that suit it.

```bash
conda env create -f aifs_ens_v2.yml          # creates the environment "aifs_ens_v2"
conda activate aifs_ens_v2
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

`flash-attn` is installed separately because it has to be built against the torch that is
already installed. It first looks for a prebuilt wheel matching the installed torch, CUDA and
Python versions. If none exists it compiles from source, which needs the CUDA toolkit (`nvcc`)
and can take an hour or more. On a cluster, load the CUDA module before installing, for example:

```bash
module load cuda/12.6        # module name differs between clusters
```

To install the environment at a specific path instead of under a name:

```bash
conda env create -f aifs_ens_v2.yml -p /path/to/envs/aifs_ens_v2
```

**Check the installation on a GPU node:**

```bash
srun -p <gpu-partition> --gres=gpu:1 --pty \
    python -c "import torch, flash_attn, anemoi.inference; print(torch.__version__, torch.cuda.is_available())"
```

This should print the torch version (`2.7.0+cu126`) and `True`. Also run `pip check`, which
reports missing or conflicting dependencies.

---

## First-time setup on a new system

### 1. Static files

These are not produced by the pipeline and must be in place before the first run.

| File | Where to get it |
|---|---|
| `weights/aifs-ens-crps-2.0.ckpt` | The `ecmwf/aifs-ens-2.0` repository on Hugging Face (Files tab). Save it under this name, or change the name in `utils/run_aifs_ens.py`. |
| `EKR/lsm.grib` | Native N320 land-sea mask, as used in ECMWF's AIFS ENS v2 example notebook. Copy it from an existing installation. |
| `EKR/mir_16_linear/*.npz` | The two earthkit-regrid interpolation matrices. Copy them from an existing installation, or generate them (see below). |

To generate the interpolation matrices (needs internet), run this in the environment. earthkit-regrid
downloads the matrices into its cache on first use:

```bash
python -c "
import numpy as np, earthkit.regrid as ekr
ekr.interpolate(np.zeros((721, 1440)), {'grid': (0.25, 0.25)}, {'grid': 'N320'})
ekr.interpolate(np.zeros(542080), {'grid': 'N320'}, {'grid': (0.25, 0.25)})
"
find ~/.cache -name "9533e90f*.npz" -o -name "7f0be51c*.npz"
```

Then copy both files into `EKR/mir_16_linear/`. The file names are hashes and must match the
names used in the scripts.

### 2. Settings to change for your system

The shell scripts contain hardcoded paths. The values shown are **examples from the original
system: change them for yours.**

**`run_aifs_pipeline.sh`**

```bash
REPO_ROOT="/net/scratch2/anustupb/aifs_v2_ens_operational"      # change
CONDA_ENV="/net/scratch2/marchakitus/conda-envs/AIFS_ENSv2"     # change (name or path)
```

**`utils/submit_aifs_v2_ens.sh`**

```bash
#SBATCH -p general                                              # change: your GPU partition
#SBATCH --gres=gpu:a100:2                                       # change: GPU type and count
#SBATCH --chdir=/net/scratch2/anustupb/aifs_v2_ens_operational  # change: repository path
REPO_ROOT="/net/scratch2/anustupb/aifs_v2_ens_operational"      # change
CONDA_ENV="/net/scratch2/marchakitus/conda-envs/AIFS_ENSv2"     # change
```

Also review the other `#SBATCH` lines (CPUs, memory, time limit) for your cluster and your
configuration.

`REPO_ROOT` is hardcoded in the job script on purpose: Slurm runs a copy of the job script from
a spool directory, so the script cannot find the repository from its own location.

If the repository path is a symbolic link, check that the compute nodes can see it. If not, use
the real path (shown by `realpath .` in the repository) in both scripts.

### 3. Make the scripts executable

```bash
chmod +x run_aifs_pipeline.sh utils/submit_aifs_v2_ens.sh
mkdir -p logs
```

---

## Running the pipeline

### The wrapper

```bash
./run_aifs_pipeline.sh                    # latest cycle available on ECMWF open data
./run_aifs_pipeline.sh --date 20260923    # a specific date (00 UTC)
```

The wrapper runs on the CPU (login node):

1. **Download.** `download_ic.py` downloads the initial conditions and reports the cycle it used.
   Without `--date` it picks the latest available date.
2. **Preprocess.** `preprocess_ic.py` builds the model input state.
3. **Inference.** Submits `utils/submit_aifs_v2_ens.sh` with `sbatch --wait` and **waits** for
   the GPU job to finish.
4. **Postprocess.** `postprocess.py` runs only if the GPU job succeeded.

It prints nothing to the terminal; follow progress in `logs/AIFS.log`.

**Keep it running.** Because the wrapper waits for the GPU job, which can take hours including
queue time, it must not be tied to your terminal session. Either run it from cron (below) or start
it in the background:

```bash
nohup ./run_aifs_pipeline.sh --date 20260923 >/dev/null 2>&1 &
```

`tmux` or `screen` also work. If the wrapper is killed anyway, the GPU job keeps running; run the
wrapper again and it will wait for that job instead of submitting a second one.

**Reruns are safe.** Every step skips work that is already done:

| Already exists | What a rerun does |
|---|---|
| IC files in `IC_data/YYYYMMDD/` | download skipped |
| `input_state_YYYYMMDDT00_v2.pkl` | preprocessing skipped |
| `Outputs/raw/init_YYYYMMDDT00.zarr` | preprocessing and GPU job skipped, goes to postprocessing |
| GPU job for that cycle queued or running | waits for it instead of submitting another |
| `Outputs/postprocessed/init_YYYYMMDDT00.nc` | nothing to do |

To redo a step, delete its output (and the outputs of the later steps) and run again.

Only one wrapper runs at a time. A second one exits immediately with code 4.

### Automatic daily runs (cron)

```bash
crontab -e
```

```cron
# m  h  dom mon dow  command
  0  8   *   *   *   /path/to/aifs_v2_ens_operational/run_aifs_pipeline.sh
```

Choose a time a few hours after 00 UTC, when the day's data is usually published. Cron uses the
server's local time zone. If the data is not out yet, the run exits with code 3 and the next
scheduled run tries again.

### Running steps by hand

Useful for debugging. Run from the repository root with the environment active:

```bash
python utils/download_ic.py --date 20260923           # prints 20260923T00 on success
python utils/preprocess_ic.py --date 20260923T00
sbatch -J aifs_v2_ens_20260923T00 --export=ALL,DATE=20260923T00 utils/submit_aifs_v2_ens.sh
python utils/postprocess.py --date 20260923T00
```

Check each result with `echo $?` (see [exit codes](#logs-and-exit-codes)).

---

## Configuration

Settings are in the **"Modify this part according to your needs"** block at the top of each
script.

| Setting | File | Notes |
|---|---|---|
| `N_MEMBERS` | `utils/run_aifs_ens.py` | ensemble size |
| `LEAD_TIME_HOURS` | `utils/run_aifs_ens.py` | forecast length, a multiple of 6 |
| `SAVE_FIELDS` | `utils/run_aifs_ens.py` | model variables to store, see below |
| `CHECKPOINT_PATH` | `utils/run_aifs_ens.py` | model weights |
| `LAT_MIN`, `LAT_MAX`, `LON_MIN`, `LON_MAX` | `utils/postprocess.py` | region; longitudes 0–360 |
| `UNIT_CONVERSIONS` | `utils/postprocess.py` | per-variable conversion (e.g. `tp` m → mm) |
| `SOURCES` | `utils/download_ic.py` | download mirrors and their order |
| `TIMEOUT`, `RETRIES`, `RETRY_WAIT` | `utils/download_ic.py` | network behaviour |
| `#SBATCH` lines | `utils/submit_aifs_v2_ens.sh` | GPU job resources |

**Choosing variables.** `SAVE_FIELDS` uses the model's output names: surface fields by short name
(`tp`, `2t`, `msl`, `10u`, `10v`), pressure-level fields as `param_level` (`u_850`, `z_500`,
`t_700`). If a name is wrong, `run_aifs_ens.py` stops and logs the full list of available names.
`postprocess.py` needs no change: it clips and saves every variable in the raw file.

Changes apply to cycles run afterwards. To rebuild an existing cycle with new settings, delete its
raw `.zarr` and postprocessed `.nc` and run again.

**Paths through environment variables.** The Python scripts accept these optional overrides:

| Variable | Default |
|---|---|
| `AIFS_CHECKPOINT` | `weights/aifs-ens-crps-2.0.ckpt` |
| `AIFS_REGRID_MATRIX` | `EKR/mir_16_linear/9533e90f…npz` |
| `AIFS_REGRID_MATRIX_N320_LATLON` | `EKR/mir_16_linear/7f0be51c…npz` |
| `AIFS_LSM_GRIB` | `EKR/lsm.grib` |
| `AIFS_DATA_ROOT` | repository root (location of `IC_data/`, `Outputs/`, `logs/`) |

The wrapper and job script expect the data folders inside the repository, so use
`AIFS_DATA_ROOT` only when running the Python scripts by hand.

---

## Outputs

### `Outputs/postprocessed/init_YYYYMMDDT00.nc`

The regional forecast, one file per cycle.

| | |
|---|---|
| Dimensions | `time` (initialization), `number` (member), `prediction_timedelta` (lead time), `lat`, `lon` |
| Grid | regular 0.25° lat/lon, latitude descending |
| Lead times | every 6 h from +6 h; the initial state is not stored |
| `tp` | **mm**, precipitation accumulated over the 6 h ending at each lead time |
| Other variables | model units (K, Pa, m s⁻¹, m² s⁻² for geopotential) unless set in `UNIT_CONVERSIONS` |

```python
import xarray as xr

ds = xr.open_dataset("Outputs/postprocessed/init_20260923T00.nc")

ens_mean = ds.tp.mean("number")                          # ensemble mean, 6-hourly
daily = ds.tp.coarsen(prediction_timedelta=4, boundary="trim").sum()  # daily totals (00-24 UTC)
total = ds.tp.sum("prediction_timedelta")                # total over the whole forecast
p_heavy = (daily > 64.5).mean("number") * 100            # % of members above 64.5 mm/day
```

### `Outputs/raw/init_YYYYMMDDT00.zarr`

The global forecast on the same 0.25° grid, before clipping and unit conversion (Zarr v2). It is
the input to `postprocess.py`. Delete old raw stores once they are postprocessed to save space.

### `IC_data/YYYYMMDD/`

The four downloaded GRIB files and the model input state. They can be deleted after the forecast
has run.

---

## Logs and exit codes

- **`logs/AIFS.log`**: every script and the wrapper write here, one line per event, with
  timestamps and the script name. Errors include the full traceback.
- **`logs/slurm-aifs_v2_ens_<cycle>-<jobid>.out`**: the GPU job's own output (GPU check, start
  and end time, exit code).

Useful commands:

```bash
tail -f logs/AIFS.log                               # follow a run
grep -E "ERROR|WARNING" logs/AIFS.log | tail        # recent problems
squeue -u $USER                                     # GPU job state
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed
```

All scripts use the same exit codes:

| Code | Meaning |
|---|---|
| 0 | success, or the output already exists |
| 1 | the step failed (network, processing, GPU job); see `AIFS.log` |
| 2 | bad arguments (e.g. wrong date format) |
| 3 | input not available: data not published yet, or an earlier step's output is missing |
| 4 | wrapper only: another pipeline run is already in progress |

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `429 Too Many Requests` or `503 Slow Down` during download | The server is busy. `download_ic.py` retries and falls back to the other mirrors (AWS, Google, ECMWF). If all fail, run again later. |
| Download exits 3 for today's date | The data is not published yet. Run later or use yesterday's date. |
| Download exits 3 for an old date | The date is no longer on the servers. |
| `Missing variables in input fields: ['w_…', 'q_50']` | The input state was made by an older preprocessing version. Delete `input_state_…_v2.pkl` and rerun `preprocess_ic.py`. |
| `Required file not found or not readable` | A script is missing or misnamed. The job script must be `utils/submit_aifs_v2_ens.sh`. |
| `sbatch: error: Unable to open file` | The path to the job script in `run_aifs_pipeline.sh` is wrong. |
| GPU job fails immediately with a `chdir` error in its `.out` file | Compute nodes cannot see the repository path (often a symlink). Use the real path in `--chdir` and `REPO_ROOT`. |
| `CUDA out of memory` | Reduce the variables saved, or use GPUs with more memory. |
| `flash_attn` import error | flash-attn is missing or was built for another torch/CUDA version. Reinstall it with the command in [Setting up the environment](#setting-up-the-environment). |
| A rerun does nothing | The output already exists. Delete it to force the step to run again. |
| `NaNs found in …` and `Coupled forcings are not supported` warnings | Expected. The NaNs are the land and sea masks of the wave and soil fields. |

---

## Scientific notes and limitations

- **All members start from the same initial state**, the IFS HRES analysis from ECMWF open data.
  The spread comes only from the model's internal noise, not from perturbed initial conditions as
  in ECMWF's operational AIFS ENS. Expect the ensemble to be under-dispersive, especially in the
  first days.
- **Reproducible noise.** Each member's random seed is derived from the date and member number,
  so rerunning a cycle reproduces it (up to GPU non-determinism), and different cycles get
  different noise.
- **Designed for medium range.** AIFS ENS v2 was trained and evaluated for forecasts up to about
  15 days. Longer lead times are an extrapolation and should be checked for drift and loss of spread.
- **Not identical to ECMWF's AIFS ENS**, because of the different initial conditions, the 0.25°
  open-data inputs regridded to N320, and GPU non-determinism.
- **Not an official ECMWF product.** The model weights are released by ECMWF under **CC BY 4.0**;
  attribution is required when you publish or share results.