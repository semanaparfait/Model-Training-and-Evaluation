# Predicting Default on SBA-Guaranteed Small-Business Loans: A Job-Creation Lens

**Author:** semana shema parfait — African Leadership University, Machine Learning summative project



## Problem

Small firms create a large share of new jobs, yet they are the borrowers banks most often turn away. Credit-guarantee schemes such as the U.S. Small Business Administration (SBA) 7(a) programme reduce the lender's risk so these firms can borrow. A default-prediction model can help a guarantor decide where to lend, but a model that is too cautious rejects repayable loans and the jobs they would have created. This project predicts whether an SBA-guaranteed loan will be **charged off (1)** or **paid in full (0)** and examines who the model wrongly flags, in particular loans to new businesses and loans that plan to create jobs.

## Dataset

U.S. SBA loan records, fiscal years 1987–2014 (about 899k rows, 27 columns), published with Li, Mickel and Taylor, "Should This Loan be Approved or Denied?", _Journal of Statistics Education_ 26(1), 2018, doi:10.1080/10691898.2018.1434342. Download `SBAnational.csv` from [Kaggle](https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied) or from the zipped copy in the Google Drive folder. The raw file is **not** committed to this repository.

## Approach

| Stage                  | What the notebook does                                                                                                                                                                                                       |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cleaning               | Keeps labelled rows, parses money strings and dates, removes post-outcome leakage columns (`ChgOffDate`, `ChgOffPrinGr`, `BalanceGross`), drops identifiers, restricts to loans approved by FY 2010 to limit right-censoring |
| Feature engineering    | Guarantee ratio, disbursed-to-approved ratio, log loan size, jobs per employee, disbursement lag, job-creation / same-state / franchise / active-during-recession flags                                                      |
| Preprocessing          | Train-only winsorising, median imputation, scaling, one-hot encoding (rare levels pooled) inside scikit-learn pipelines                                                                                                      |
| Traditional ML (E1–E6) | Logistic regression (baseline, + features, + class weights), random forest (default, tuned), histogram gradient boosting (tuned)                                                                                             |
| Deep learning (E7–E9)  | Keras Sequential MLP, Functional API model with categorical embeddings, random search over width / dropout / L2 / learning rate; all fed through `tf.data`                                                                   |
| Robustness (E10–E12)   | Decision-threshold selection, time-based split, ablation without `Term`                                                                                                                                                      |
| Evaluation             | Held-out test set, confusion matrices, ROC and precision–recall curves, learning curves, permutation importance, subgroup error analysis                                                                                     |

Every run is logged to Weights & Biases in offline mode (`outputs/wandb`) and Keras runs to TensorBoard (`outputs/tensorboard`); the experiment table is written to `outputs/results/experiment_table.csv`.

## How to run

**Colab (recommended).** Open `SBA_Job_Creation_Default_Prediction.ipynb` in Colab, upload `SBAnational.csv` through the Files panel (or leave `DATA_PATH = None` and let `kagglehub` download it), then _Runtime → Run all_. The first cell installs the few packages Colab lacks.

**Locally.**

```bash
git clone https://github.com/semanaparfait/Model-Training-and-Evaluation.git
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place SBAnational.csv in the repo root or in data/
jupyter notebook SBA_Job_Creation_Default_Prediction.ipynb
```

Set `FAST_DEBUG = True` in the configuration cell for a run of a few minutes that only checks the pipeline end to end. All randomness is seeded (`SEED = 42`).

**Viewing logs.**

```bash
tensorboard --logdir outputs/tensorboard
wandb sync outputs/wandb/offline-run-*      # optional, uploads offline runs to a W&B account
```

## Repository layout

```
├── SBA_Job_Creation_Default_Prediction.ipynb   # full pipeline, executed with outputs
├── requirements.txt
├── README.md
└── .gitignore                                  # excludes data, models and logs
```

Running the notebook creates `outputs/` with `models/`, `results/`, `figures/`, `wandb/` and `tensorboard/`.



## Limitations

U.S.-only data seen from the lender's side; outcomes of recent loans are censored; borrower characteristics such as credit history are absent; the random split overstates performance compared with a time-based split; and `Term` may act partly as a proxy for the outcome (tested in E12).

## AI-assistance statement

AI assisted me doing readme only
