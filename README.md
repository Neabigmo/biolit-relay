# BioLit-Relay

Code for criterion-aware routing in budgeted systematic-review screening.

## Requirements

- Python 3.12
- About 40 MB for the downloaded source data

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

## Download the data

The repository does not contain or redistribute the SYNERGY records. Download
the five required review folders from DataverseNL through the supplied script:

```powershell
.venv\Scripts\python.exe scripts\00_fetch_synergy.py
```

The script downloads SYNERGY v1.0 from DOI
[`10.34894/HE6NAQ`](https://doi.org/10.34894/HE6NAQ) and verifies the source
checksums. Review the dataset's terms at the source before use.

## Reproduce

Run these commands from the repository root:

```powershell
.venv\Scripts\python.exe scripts\01_verify_data.py
.venv\Scripts\python.exe scripts\02_build_specs.py
.venv\Scripts\python.exe scripts\03_run_baselines.py --n-jobs 4
.venv\Scripts\python.exe scripts\04_run_biolit_relay.py --n-jobs 4
.venv\Scripts\python.exe scripts\05_evaluate.py
.venv\Scripts\python.exe scripts\06_make_reports.py
.venv\Scripts\python.exe -m pytest -q
```

Downloaded data and generated results are written under `data/` and `outputs/`;
both directories are excluded from Git.
