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
├── config.yaml                 # Configurable settings
├── main.py                     # Entry point: runs the complete pipeline
├── requirements.txt           # Python dependencies
│
├── Classes/                    # Course/lab notebooks organized by week
│   ├── W3_Notebooks/
│   │   
│   └── W4_Notebooks/
│
├── data/
│   ├── compas_two_year_recidivism.csv
│   └── README.md              # Dataset description and data dictionary
│
├── results/                   # Generated results from pipeline runs
│
├── src/
│   ├── __init__.py
│   ├── data.py                # Data loading
│   ├── data_diagnostics.py    # Data quality and diagnostic checks
│   ├── evaluate.py            # Model evaluation and fairness checks
│   ├── model.py               # Model construction
│   ├── preprocessing.py       # Data cleaning and preprocessing pip
│   └── results.py             # Saves results from each run
│
└── venv/                      # Virtual environment (ignored by Git)
```

## Pipeline Progress

The pipeline is being improved incrementally throughout the practical classes.

| Week  | Focus                           | Main changes                                                                                                                                                                                                                                             |
| ----- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2** | Introduction and baseline         | Created the initial end-to-end pipeline, including data loading, basic preprocessing, train/test split, Logistic Regression baseline, accuracy evaluation, train/test comparison, and an initial fairness check using false-positive rates by race.      |
| **3** | EDA, preprocessing and validation | Added data diagnostics, improved missing-value handling, domain-rule checks, duplicate removal, category standardisation, leak-safe preprocessing, redundant-feature removal, empirical preprocessing selection, and 5-fold stratified cross-validation. |
| **4** | Model comparison and validation | Changed the selected scaler from Robust Scaling to Standard Scaling, evaluated Decision Tree, Random Forest, Logistic Regression and a Dummy baseline using the same 5-fold stratified cross-validation procedure, and extended the fairness analysis to out-of-fold predictions across the full development set. |

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

### Week 4 - Model Comparison and Cross-Validation

Week 4 extended the evaluation beyond a single model and made cross-validation the main basis for comparing the candidate models.

The main changes were:

* Changed the selected scaler from Robust Scaling to Standard Scaling.
* Evaluated four models using the same preprocessing and validation procedure: Decision Tree, Random Forest, Logistic Regression, and Dummy Classifier baseline
* Used 5-fold stratified cross-validation with accuracy as the main comparison metric.
* Reported the mean and standard deviation of training and validation accuracy across folds.
* Calculated the mean train-validation gap to assess possible overfitting.
* Generated out-of-fold predictions for the complete development set.
* Extended the fairness analysis to the out-of-fold predictions rather than relying on a single train/test split.
* Kept the test set locked and unevaluated during model selection.

This change makes the model comparison less dependent on the particular rows that happen to appear in a single validation split.

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

During Week 4, the scaler used in the pipeline was changed from Robust Scaling to Standard Scaling. The results below allow the effect of this change to be assessed together with the updated cross-validation procedure.

The full comparison and paired analysis are available in `02_preprocessing.ipynb`.

## Models and Validation

From Week 4 onwards, the pipeline evaluates four models using the same development set and 5-fold stratified cross-validation:

* Decision Tree
* Random Forest
* Logistic Regression
* Dummy Classifier baseline

The Dummy Classifier uses the most_frequent strategy and provides a reference point for interpreting the performance of the trained models.

The main validation metric for Week 4 is accuracy.

For each model, the pipeline reports:

* Training accuracy for each fold
* Validation accuracy for each fold
* Mean training accuracy
* Mean validation accuracy
* Standard deviation of validation accuracy
* Mean train-validation gap
* Classification report from out-of-fold predictions
* False-positive rate by race from out-of-fold predictions

The development set contains 5,771 rows. A separate 1,443-row test set remains locked and is not used during model selection.

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

### Week 4 - Model Comparison

Week 4 uses 5-fold stratified cross-validation with accuracy as the main metric. The same validation procedure was applied to all four candidate models.

| Model | Mean Train Accuracy | Train SD | Mean Validation Accuracy | Validation SD | Mean Gap |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 68.4% | 0.5% | 67.4% | 1.8% | +1.0% |
| Random Forest | 69.0% | 0.5% | 68.0% | 1.5% | +1.0% |
| Logistic Regression | 67.5% | 0.3% | 67.2% | 1.3% | +0.3% |
| Dummy Classifier | 54.9% | 0.0% | 54.9% | 0.0% | 0.0% |

The three trained models have similar classification behaviour. In particular, all three identify class 0 more successfully than class 1, with class 1 recall ranging from 0.51 to 0.54.

The Dummy Classifier illustrates why accuracy alone should not be interpreted without considering the class distribution: although it obtains 54.9% accuracy, it predicts only class 0 and therefore has 0.00 recall for class 1.
 
### Week 3 vs Week 4

The Decision Tree provides the most direct comparison between the two weeks:

| Metric | Week 3 | Week 4 |
|---|---|---|
| CV Mean Accuracy | **67.8%** | **67.4%** |
| CV Accuracy SD | 1.3% | 1.8% |
| Validation Procedure | 5-fold stratified CV | 5-fold stratified CV |
| Scaler | Previous configuration | **Standard Scaling** |
| Models Evaluated | Decision Tree | DT, RF, LR, Dummy |

The Week 4 Decision Tree did not improve mean cross-validation accuracy, decreasing slightly from 67.8% to 67.4%. Therefore, Standard Scaling should not be described as an accuracy improvement.

Instead, Week 4 provides a more systematic comparison of four models using the same 5-fold stratified cross-validation procedure. The Random Forest achieved the highest mean validation accuracy at 68.0%, but the differences between the Decision Tree, Logistic Regression, and Random Forest were small.

Overall, Week 4 suggests that predictive accuracy remained around 67–68%. The main improvement was the more reliable and fair evaluation using 5-fold cross-validation and Out-of-Fold predictions, showing that the baseline results were stable rather than dependent on a single data split.

## Fairness Evaluation

Fairness is evaluated using **false-positive rate (FPR) by race**.

The analysis compares the model's FPR with the corresponding FPR observed for COMPAS.

For Week 4, FPR is calculated from out-of-fold predictions on the complete 5,771-row development set. This means that every development observation receives a prediction from a model that was trained without that observation being included in the corresponding training fold.

This provides a more robust fairness evaluation than calculating the metric from a single validation split.

The results should still be interpreted carefully, particularly for groups with very small sample sizes.

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
5. Perform 5-fold stratified cross-validation on the development set.
6. Train and evaluate the configured models.
7. Generate out-of-fold predictions.
8. Perform the fairness analysis.
9. Refit the configured final model on the complete development set.
10. Save the results to results/.

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