# ETAI - COMPAS Recidivism Prediction

**Maria Inês Lopes dos Santos, 20231630**

## Project Overview

This project develops a predictive pipeline for **two-year recidivism** using ProPublica's COMPAS dataset.

The dataset comes from the data used in ProPublica's 2016 investigation into COMPAS, a risk-assessment algorithm used in the US criminal justice system to help inform decisions such as bail and sentencing.

The project started from a simple baseline pipeline provided for the practical classes. It is being progressively improved throughout the semester by addressing weaknesses in the original preprocessing, validation, and evaluation procedures.

The main goal is not only to build a predictive model, but also to make the pipeline **reliable, reproducible, and easier to evaluate for both predictive performance and fairness**.

## Project Structure

```text
.
├── .gitignore
├── config.yaml                   # Configurable settings
├── main.py                      # Entry point: runs the complete pipeline
├── requirements.txt             # Python dependencies
│
├── data/
│   ├── compas_two_year_recidivism.csv
│   └── README.md                # Dataset description and data dictionary
│
├── results/                     # Generated results from pipeline runs
│
├── src/
│   ├── __init__.py
│   ├── data.py                  # Data loading
│   ├── data_diagnostics.py      # Data quality and diagnostic checks
│   ├── evaluate.py              # Model evaluation and fairness checks
│   ├── model.py                 # Model construction
│   ├── preprocessing.py         # Data cleaning and preprocessing pipeline
│   └── results.py               # Saves results from each run
│
├── venv/                        # Virtual environment (ignored by Git)
│
└── W3_Notebooks/                # Exploratory notebooks and lab materials
    ├── data/
    ├── 01_eda_introduction.ipynb
    └── 02_preprocessing.ipynb
```

## Pipeline Progress

The pipeline is being improved incrementally throughout the practical classes.

| Week  | Focus                           | Main changes                                                                                                                                                                                                                                             |
| ----- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2** | Introduction and baseline         | Created the initial end-to-end pipeline, including data loading, basic preprocessing, train/test split, Logistic Regression baseline, accuracy evaluation, train/test comparison, and an initial fairness check using false-positive rates by race.      |
| **3** | EDA, preprocessing and validation | Added data diagnostics, improved missing-value handling, domain-rule checks, duplicate removal, category standardisation, leak-safe preprocessing, redundant-feature removal, empirical preprocessing selection, and 5-fold stratified cross-validation. |

### Week 2 - Baseline

The initial pipeline was deliberately simple and provided a starting point for the project.

It included:

* A single train/test split.
* Basic preprocessing using `dropna()` and one-hot encoding.
* Logistic Regression as the baseline model.
* Train and test accuracy comparison to identify possible overfitting.
* A basic fairness analysis comparing the model's false-positive rate by race with COMPAS.
* Automatic saving of results to the `results/` directory.

### Week 3 - EDA, Preprocessing and Validation

The pipeline was substantially improved based on the findings from the EDA.

The main changes were:

* Diagnosed missingness using statistical tests.
* Checked for invalid values using domain rules.
* Identified and removed duplicate rows.
* Standardised inconsistent categorical values.
* Added missingness indicators where missingness was informative.
* Replaced the previous `dropna()` approach with explicit imputation.
* Moved preprocessing into a leak-safe pipeline applied after the train/test split.
* Identified and removed redundant features using correlation and VIF.
* Empirically compared different encoder/scaler combinations.
* Added 5-fold stratified cross-validation to assess model stability.

## Preprocessing

The preprocessing decisions were based on the EDA performed in:

* `Practical/W3/notebooks/01_eda_introduction.ipynb`
* `Practical/W3/notebooks/02_preprocessing.ipynb`

### Missing and Invalid Values

| Column(s)                                              | Issue                                     | Treatment                                 |
| ------------------------------------------------------ | ----------------------------------------- | ----------------------------------------- |
| `age`                                                  | ~2% missing                               | Median imputation                         |
| `juv_fel_count`                                        | ~3% missing                               | Median imputation                         |
| `priors_count`                                         | ~7% missing, including placeholder values | Median imputation + missingness indicator |
| `c_charge_degree`                                      | ~3.2% missing                             | Mode imputation + missingness indicator   |
| `race`                                                 | ~1% missing / placeholder values          | Mode imputation                           |
| `sex`                                                  | ~1.5% missing / placeholder values        | Mode imputation                           |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | Invalid or out-of-range values            | Converted to `NaN` before imputation      |

Missingness in `priors_count` and `c_charge_degree` was treated differently because the EDA indicated that it was associated with other variables, particularly `age_cat`. Missingness indicators were therefore added for these columns.

### Categorical Values

Inconsistent category values caused by differences in casing, whitespace, or abbreviations were standardised so that each category has a consistent representation.

### Duplicate Rows

The dataset contained **72 exact duplicate rows**, all associated with repeated IDs.

These duplicates were removed while keeping the first occurrence.

### Redundant Features

The following features were removed because they contained information already represented by other variables:

* `prior_offenses`
* `age_in_months`
* `juvenile_total`

The redundancy was identified using correlation analysis and VIF. In particular, `juvenile_total` was identified as an exact sum of other variables through the multicollinearity analysis.

## Encoder and Scaler Selection

Different preprocessing combinations were compared empirically using Logistic Regression.

The experiment evaluated:

* **4 encoders:** one-hot, ordinal, count, and target encoding.
* **4 scalers:** none, standard, min-max, and robust scaling.
* **15 repeated train/test splits.**

The combinations were compared using mean accuracy across the repeated splits.

The best-performing combination in this experiment was:

**Target Encoding + Standard Scaling**

However, the paired comparison with the runner-up showed that the difference was within the variation observed across the repeated splits. Therefore, the selected preprocessing combination should not be interpreted as definitively superior.

The full comparison and paired analysis are available in `02_preprocessing.ipynb`.

## Model and Validation

The current pipeline uses a **Decision Tree** model.

The model is evaluated using:

* Test accuracy
* Balanced accuracy
* ROC-AUC
* Classification report
* Train/test performance comparison
* 5-fold stratified cross-validation

The use of cross-validation provides a more reliable estimate of model stability than relying only on a single train/test split.

## Results

### Week 2 - Baseline Results

Two baseline models were evaluated:

| Model               | Test Accuracy | Train–Test Gap |
| ------------------- | ------------: | -------------: |
| Logistic Regression |     **67.8%** |       **0.1%** |
| Decision Tree       |         66.8% |           1.2% |

The models had similar performance on the test set.

The Decision Tree had higher recall for class 1:

* Decision Tree: **0.65**
* Logistic Regression: **0.60**

The F1-scores were also similar:

* Class 0: Logistic Regression **0.72**, Decision Tree **0.69**
* Class 1: Logistic Regression **0.63**, Decision Tree **0.64**

For the main African-American group, the false-positive rates were:

* Logistic Regression: **0.33**
* Decision Tree: **0.39**
* COMPAS: **0.44**

These results were based on the specific configurations tested and should not be interpreted as the maximum performance achievable by either model, since no hyperparameter search was performed at this stage.

### Week 3 - Improved Pipeline

With the improved preprocessing pipeline, the Decision Tree achieved:

* **66.5% test accuracy**
* **65.0% balanced accuracy**
* **0.706 test ROC-AUC**

Using 5-fold stratified cross-validation:

* **67.8% mean accuracy**
  SD = **1.3%**
* **0.716 mean ROC-AUC**
  SD = **1.3%**

### Week 2 vs. Week 3

The Week 3 Decision Tree achieved a very similar test accuracy to the Week 2 Decision Tree:

| Metric            | Week 2 | Week 3 |
| ----------------- | -----: | -----: |
| Test Accuracy     |  66.8% |  66.5% |
| Balanced Accuracy |      — |  65.0% |
| Test ROC-AUC      |      — |  0.706 |
| CV Mean Accuracy  |      — |  67.8% |
| CV Accuracy SD    |      — |   1.3% |
| CV Mean ROC-AUC   |      — |  0.716 |
| CV ROC-AUC SD     |      — |   1.3% |

The small difference in test accuracy suggests that the new preprocessing did not produce a clear improvement in accuracy on this particular test split.

However, Week 3 represents an important methodological improvement. The pipeline now handles missing and invalid data more carefully, avoids preprocessing leakage, removes redundant features, and uses cross-validation to assess model stability.

Therefore, the main progress from Week 2 to Week 3 is **the quality and reliability of the pipeline and validation process**, rather than a substantial increase in predictive accuracy.


## Fairness Evaluation

Fairness is evaluated using **false-positive rate (FPR) by race**.

The analysis compares the model's FPR with the corresponding FPR observed for COMPAS on the test data.

This provides an initial view of whether the model's errors differ across racial groups.

Fairness analysis is treated as a separate part of the evaluation rather than being reduced to a single overall performance metric. Further analysis may be added as the project develops.

## Running the Pipeline

### 1. Create the environment

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Windows — PowerShell

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

#### Windows — Command Prompt

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

The virtual environment only needs to be created and configured once.

### 2. Run the project

Every subsequent time, activate the environment and run:

#### macOS / Linux

```bash
source venv/bin/activate
python main.py
```

#### Windows

```powershell
venv\Scripts\activate
python main.py
```

The pipeline will:

1. Load the dataset.
2. Run data diagnostics.
3. Clean and preprocess the data.
4. Split the data into training and test sets.
5. Train the model.
6. Evaluate predictive performance.
7. Perform the fairness analysis.
8. Save the results to `results/`.

## Troubleshooting

### PowerShell execution policy

If PowerShell blocks the virtual-environment activation script, the following command can be used for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

For a persistent user-level setting:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

If this is blocked by a school or organisation policy, Git Bash or Command Prompt can be used instead.

### Windows file permissions

If Python or the terminal cannot read or write files inside `Documents`, `Desktop`, or `Pictures`, check:

* **Windows Security → Virus & threat protection → Manage ransomware protection → Controlled folder access**
* **Settings → Privacy & security → File system**

Alternatively, move the project to another directory where the current user has full read/write access.

## Saving and Comparing Results

Each pipeline run is automatically saved as a timestamped file in:

```text
results/
```

For example:

```text
results/run_20260916_143012.txt
```

This makes it possible to compare results across different practical classes and model configurations.

The `results/` directory is generated automatically and is not tracked by Git because it contains generated output rather than source code.

## GitHub Workflow

From the project root, with the virtual environment active:

```bash
git add .
git commit -m "Short description of changes"
git push
```

If GitHub asks for a password when using HTTPS, GitHub no longer accepts the normal account password for Git authentication. A Personal Access Token (PAT) or SSH authentication can be used instead.


## Dataset

For the dataset description, problem definition, and complete data dictionary, see:

```text
data/README.md
```

## Project Status

**Current stage: Week 3**

The project has progressed from a simple baseline predictive pipeline to a more robust pipeline with:

* Data diagnostics
* Improved missing-value handling
* Invalid-value detection
* Category standardisation
* Duplicate removal
* Redundant-feature removal
* Leak-safe preprocessing
* Empirical preprocessing selection
* Decision Tree modelling
* 5-fold stratified cross-validation
* Predictive performance evaluation
* Initial fairness evaluation

Future practical classes will continue to build on this pipeline.





