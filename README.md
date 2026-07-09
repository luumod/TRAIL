# TRAIL

This repository contains the anonymized source code for the BIBM submission.
It includes model code, training/evaluation entry points, and data preprocessing
scripts only. Processed datasets, raw MIMIC files, checkpoints, logs, and
hyperparameter-search artifacts are intentionally not included.

## Expected Data Layout

After preparing MIMIC-III or MIMIC-IV, place the processed files under:

```text
data/output/mimic-iii/
  records_final.pkl
  voc_final.pkl
  ehr_adj_final.pkl
  ddi_A_final.pkl
  ddi_mask_H.pkl
  atc3toSMILES.pkl

data/output/mimic-iv/
  records_final.pkl
  voc_final.pkl
  ehr_adj_final.pkl
  ddi_A_final.pkl
  ddi_mask_H.pkl
  atc3toSMILES.pkl
```

The preprocessing scripts are provided in `data/processing-iii.py` and
`data/processing-iv.py`. Users need to obtain access to MIMIC and generate the
processed files locally.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Some causal-discovery components depend on CDT/GES and may require an R setup
compatible with CDT.

## Training

```bash
python src/main_train.py --dataset mimic-iii --device cuda:0 --epochs 30 --commit mimiciii_run
```

For MIMIC-IV:

```bash
python src/main_train.py --dataset mimic-iv --device cuda:0 --epochs 30 --commit mimiciv_run
```

To evaluate a saved checkpoint:

```bash
python src/main_train.py --dataset mimic-iii --test --resume_path path/to/checkpoint.model
```

The public entry point exposes only essential runtime options. Paper/default
model hyperparameters are fixed inside the code to keep the release concise.
