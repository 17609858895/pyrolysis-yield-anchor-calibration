# Local calibration of biomass pyrolysis yields

Analysis code for *Local calibration improves bio-oil yield estimates and temperature selection under a biochar constraint*. The models use published feedstock and operating conditions to predict bio-oil and biochar yields. For an unseen experimental campaign, one or more measured runs can calibrate the predictions for the remaining runs. The decision analysis evaluates temperature ranking, attainable oil yield and selection under a minimum biochar yield.

## Input data

Download `SI_Data.xlsx` from [Shayan Abaei's pyrolysis data, Zenodo v2](https://doi.org/10.5281/zenodo.13645157), which are distributed under CC BY 4.0. Place the original workbook in `raw/SI_Data.xlsx` at the repository root. Its SHA-256 is `7cb3354c4a3143cd3f30b3614ae2302d604937894566127b7e5b7c14074fc038`. The source's companion reference workbook identifies the contributing publications.

## Run

Use Python 3.12 and run the commands below from the repository root. The study used PyTorch 2.13.0 on CPU. Tables and predictions are written under `tables/` and `results/`.

```sh
python -m pip install -r requirements.txt
python scripts/01_clean_data.py
python scripts/12_lab_fingerprint.py
```

Cleaning yields 1,667 observations, with reported biochar yield for 1,544 and bio-oil yield for 1,331. The fingerprint analysis evaluates classification of the 102 source studies with at least five observations. It provides a quick check of the data and grouping pipeline.

For the calibration and decision results, run each of the three commands in the next block for **each seed 3, 4 and 5**, replacing `3` in the example. Each partition holds out complete connected groups of source studies and matching feedstock profiles. Fitting the models can take substantial CPU time.

```sh
python scripts/04_kshot_eval.py 3 confirm
python scripts/11_literature_scaling.py 3
python scripts/13_decision_value.py 3
```

Once all three seeds have finished, summarise the held-out predictions.

```sh
python scripts/06_kshot_summary.py confirm
python scripts/11b_summarise_scaling.py
python scripts/13b_decision_summary.py
python scripts/16_matched_anchor_comparison.py
```

`T06_kshot_metrics_confirm.csv` reports each model's error by product and anchor count. `T12_literature_scaling.csv` compares training-set sizes. Tables T16–T19 report decision results, and `T23_matched_anchor_comparison.csv` compares zero versus one anchor on identical query observations, with bootstrap intervals over source groups. Yield errors are absolute percentage points of reported product mass yield. In the model outputs, `HTS` denotes the hierarchical thermal-survival model, `CB-ME` denotes mixed-effects CatBoost, and `HTS+CB-ME` denotes their prediction average.

The estimator and its boosting alternative are in `scripts/hts.py` and `scripts/hts_boost.py`. The predictor definitions, fold construction and scoring functions are in `scripts/common.py`. The conventional models and stacked benchmark can be run separately after cleaning:

```sh
python scripts/02_benchmark_20.py
python scripts/08_stacking.py
```
