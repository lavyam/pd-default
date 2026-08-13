# PD Model – Script Overview

This repository contains the code for training and running a Probability of Default (PD) model.  
Below is a short description of each script file.

---

## 1. estimation.py
Contains all model-estimation logic:
- cleans the raw training data
- engineers financial ratios
- fits the logistic/XGBoost PD model
- performs calibration (bucket table)
- saves/loads model artifacts

The name *estimation* reflects that this script performs the end-to-end model estimation process.

---

## 2. predict.py
Contains only prediction logic:
- loads saved artifacts
- applies data cleaning + ratio engineering
- outputs PD predictions including calibration for new observations

This script’s sole responsibility is scoring new data.

---

## 3. harness.py
Command-line interface (CLI) "harness" that ties everything together:
- `--train` trains the model and saves artifacts
- `--predict` scores new applications and outputs calibrated PDs

It acts as a wrapper over the entire workflow.

---

## 4. requirements.txt
Lists Python dependencies required to run the PD model 
(e.g., numpy, pandas, scikit-learn).

---

### 5. save_artifacts.py:
Runs the training pipeline and saves the model artifacts

---

## 6. Plots

### Plots for Data Cleaning and Understanding

These plots compare the default rate (`default_next_year`) against quantile bins of each financial ratio.

For every variable:

- The data is split into 10 equal-sized bins (deciles).
- We compute the mean default rate inside each bin.
- We plot how default probability changes across the bins.

This lets us see:

- Whether the relationship is monotonic (e.g., defaults decrease steadily as profitability increases)
- Whether it is U-shaped or non-linear
- Whether certain variables have weak or no predictive power
- How well each variable separates good firms from risky firms


### Plots for Callibration and Power


#### Calibration Related
Contains:
•⁠  ⁠Calibration bucket tables
•⁠  ⁠Bar-at-X calibration plot (Empirical vs Calibrated PD)

What it shows:
•⁠  ⁠Predicted PDs are grouped into buckets (typically deciles)
•⁠  ⁠For each bucket, we compare:
  - *Calibrated PD (model output)*
  - *Empirical PD (actual default rate)*

A 45-degree dashed line represents perfect calibration.

If bars track this line → the model’s predicted PDs match reality.

*Purpose:*  
Checks whether the model *produces PDs that reflect true default frequencies*.


#### Power Related



Contains:
•⁠  ⁠ROC curve (Receiver Operating Characteristic) after calibration
•⁠  ⁠AUC score (Area Under Curve)

What it shows:
•⁠  ⁠Ability of the model to rank high-risk firms above low-risk firms

*Purpose:*  
Measures the *discriminative power* of the PD model.



## Running the Model

### Train: 
Done using train-2025.csv


### Harness: 
To test on a new test dataset, run the harness function with
python3 harness.py --input_csv test.csv --output_csv results_/output_csv

Where you can replace test.csv with your test dataset and results_/output_csv with your preferred destination path.


### Data Leakage:

The project contained 2 sources of data leakage. Which caused the model to use data from future instead of the info available at the scoring time.

1. Leakage in Fallback Imputation (Latent Factors)
    Description: The fallback imputation computed means, standard deviations, and factor loadings using all available data, including new data (test). This allowed the imputer to "peek" into the future.
    Fix:
    1. Fit the imputer only on the training data.
    2. Store all params such as means, stds, factor loadings inside `artifacts.factor_imputer` 
    3. During inference, use frozen imputer that doesn't recompute anything

2. Leakage in Growth Ratios Calculation
    Description: Growth ratios were calculated by `pct_change`. This means the growth values depended on other rows inside the same scoring file, which is invalid.
    Example: In the test data, if there is only 1 record of firm => growth = NaN, but if there are multiple records => growth != NaN
    This shouldn't be the case, as growth should be calculated based on historically available data instead of current data/peeking into future

    Fix:
    1. Precompute historical values i.e. last known financial statement per borrower, on training data only
    2. Store them in:
      `artifacts.history_table`
    3. During inference, compute growth as the difference between the current value and the last known "historic value"

Verification:
`test.py` checks if a record receives the same P.D if it's scored alone or with a group of data. It prints out the PDs and the absolute difference between both. If abs = 0, the model is invariant to new data.
