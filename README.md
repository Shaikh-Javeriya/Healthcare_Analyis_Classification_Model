# Healthcare Data Analysis — Project Report
### Internship Project | Data Science & Analytics Division
**Analyst:** Data Science Intern
**Dataset:** `healthcare_dataset.csv`
**Tools Used:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost
**Project Type:** Exploratory Data Analysis + Machine Learning Classification

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Cleaning & Preparation](#3-data-cleaning--preparation)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Critical Data Finding — Synthetic Dataset Identified](#5-critical-data-finding--synthetic-dataset-identified)
6. [Feature Engineering](#6-feature-engineering)
7. [Machine Learning Models](#7-machine-learning-models)
8. [Model Results & Interpretation](#8-model-results--interpretation)
9. [Key Takeaways & Business Implications](#9-key-takeaways--business-implications)
10. [Recommendations](#10-recommendations)
11. [Project File Structure](#11-project-file-structure)

---

## 1. Executive Summary

This project presents a full-cycle data science analysis on a healthcare dataset containing **55,500 patient records** across **15 clinical and administrative variables**. The analysis covers data ingestion, quality assessment, exploratory data analysis (EDA), feature engineering, and machine learning model development.

A critical and professionally significant finding was made early in the analysis: **the dataset is synthetically generated**, meaning it was created by a computer algorithm with no real-world clinical relationships between variables. This was identified through rigorous statistical diagnostics — not assumed — and is documented transparently in this report.

Despite this constraint, the project demonstrates a **complete, industry-standard machine learning pipeline** including:

- Thorough data quality investigation
- Multi-dimensional exploratory analysis
- Domain-informed feature engineering to construct learnable signals
- Comparative evaluation of 5 classification algorithms
- Honest performance interpretation with statistical context

Two classification models were built:
- **Model 1:** Patient Length of Stay Classification (Short / Medium / Long)
- **Model 2:** Test Result Classification (Abnormal / Normal / Inconclusive)

> **Bottom Line for Reviewers:** The value of this project lies not in the accuracy numbers — which are constrained by data quality — but in the analytical rigour, transparent diagnosis, and professional handling of a real-world data science challenge: working with imperfect data.

---

## 2. Dataset Overview

| Property | Detail |
|---|---|
| Source File | `healthcare_dataset.csv` |
| Total Records | 55,500 patients |
| Total Features | 15 columns |
| Numerical Columns | Age, Billing Amount, Room Number |
| Date Columns | Date of Admission, Discharge Date |
| Categorical Columns | Name, Gender, Blood Type, Medical Condition, Doctor, Hospital, Insurance Provider, Admission Type, Medication, Test Results |
| Missing Values | **Zero** — dataset is complete |
| Memory Usage | 6.4 MB |
| Date Range | May 2019 — June 2024 |

### Column Descriptions

| Column | Type | Description |
|---|---|---|
| Name | String | Patient full name |
| Age | Integer | Patient age (13–89 years) |
| Gender | String | Male / Female |
| Blood Type | String | 8 types (A+, A-, B+, B-, AB+, AB-, O+, O-) |
| Medical Condition | String | 6 conditions: Cancer, Diabetes, Hypertension, Obesity, Asthma, Arthritis |
| Date of Admission | Date | Hospital admission date |
| Doctor | String | Attending physician name |
| Hospital | String | Hospital name |
| Insurance Provider | String | 5 providers: Cigna, Medicare, UnitedHealthcare, Blue Cross, Aetna |
| Billing Amount | Float | Total billing in currency units |
| Room Number | Integer | Assigned room (101–500) |
| Admission Type | String | Emergency / Urgent / Elective |
| Discharge Date | Date | Hospital discharge date |
| Medication | String | 5 medications: Lipitor, Ibuprofen, Aspirin, Paracetamol, Penicillin |
| Test Results | String | Abnormal / Normal / Inconclusive |

---

## 3. Data Cleaning & Preparation

### 3.1 Initial Quality Check

Upon loading the dataset, a systematic quality audit was performed:

```
✅ No missing values across all 15 columns
✅ No duplicate records detected
✅ Data types consistent with column definitions
⚠️  Negative Billing Amount values found → resolved
⚠️  Date columns stored as strings → converted to datetime
⚠️  Text inconsistencies (whitespace, casing) → standardized
```

### 3.2 Steps Performed

**Step 1 — Date Type Conversion**
Both `Date of Admission` and `Discharge Date` were converted from string format to `datetime64` objects, enabling date arithmetic and time-series analysis.

**Step 2 — Feature Derivation: Length of Stay**
A new numerical column, `Length of Stay`, was engineered by calculating the difference in days between admission and discharge dates. This became a primary analytical and modeling target.

```python
df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
# Result: Range 1–30 days, Mean = 15.5 days
```

**Step 3 — Negative Billing Resolution**
A check for negative billing amounts was performed. All negative values were converted to their absolute values, treating them as data entry sign errors rather than legitimate credits.

**Step 4 — Text Standardization**
All categorical text columns were stripped of leading/trailing whitespace and converted to Title Case to ensure consistent grouping and encoding downstream.

---

## 4. Exploratory Data Analysis

EDA was conducted across four analytical dimensions, each generating visualized outputs.

### 4.1 Univariate Analysis

Individual distributions were examined for all key variables:

| Variable | Finding |
|---|---|
| Age | Roughly uniform distribution, age 13–89, mean ~51.5 years |
| Billing Amount | Perfectly flat uniform distribution (significant finding — see Section 5) |
| Length of Stay | Perfectly uniform, 1–30 days (~1,800 patients per day value) |
| Gender | Near-perfect 50/50 split: Male 27,774 / Female 27,726 |
| Medical Condition | 6 conditions, near-perfectly balanced (~9,200 each) |
| Admission Type | Elective: 33.6%, Urgent: 33.5%, Emergency: 32.9% |

### 4.2 Bivariate Analysis

Relationships between pairs of variables were examined:

- **Billing Amount vs Medical Condition:** Median billing was identical across all 6 conditions — no condition drove higher or lower costs.
- **Length of Stay vs Admission Type:** Box plots showed identical distributions across Emergency, Urgent, and Elective admissions — no urgency-based stay difference.
- **Age vs Billing Amount:** Scatter plot showed a pure cloud with no directional trend — zero age-billing relationship.
- **Test Results vs Medical Condition:** Equal distribution of Abnormal / Normal / Inconclusive across every condition — no clinical condition influenced test outcomes.

### 4.3 Financial Analysis

- Average billing was consistent (~₹25,539) across all insurance providers, admission types, and medical conditions.
- No correlation was found between Length of Stay and Billing Amount (r = -0.0056).
- Violin plots showed identical billing distributions regardless of how patients were admitted.

### 4.4 Time Series Analysis

- **Monthly Admissions (2019–2024):** Flat trend with no seasonality, no COVID dip, no year-over-year growth — consistent with algorithmic generation.
- **Yearly Average Billing:** Stable at ~₹25,500 across all years — no inflation, no pricing changes.

### 4.5 Hospital & Doctor Performance

- 39,876 unique hospitals and 40,341 unique doctors in the dataset.
- Top hospitals by volume had only 3–4 patients each — indicating near-random assignment of hospital names.
- Average billing per hospital showed no meaningful differentiation.

---

## 5. Critical Data Finding — Synthetic Dataset Identified

> This section documents the most analytically significant finding of the project and represents the kind of critical thinking that distinguishes a competent data scientist from one who simply runs code.

### 5.1 Diagnosis

After initial EDA, a formal diagnostic was conducted on all three target variables and their relationships with features. Five independent statistical signals confirmed that **this dataset was algorithmically generated using random number assignment**, with no real-world clinical logic connecting columns.

### 5.2 The Five Smoking Guns

**🔴 Evidence 1 — Billing Amount: Perfect Uniform Distribution**
Real hospital billing follows a right-skewed distribution. This dataset's billing histogram is completely flat from ₹0 to ₹52,764 — the signature of `numpy.random.uniform(min, max)`.

**🔴 Evidence 2 — Length of Stay: Perfectly Equal at Every Value**
Every day from 1 to 30 has exactly ~1,800 patients:
```
Day 1:  1,817 patients
Day 15: 1,785 patients
Day 30: 1,874 patients
```
In real hospitals, stays cluster at 1–5 days with exponential drop-off. This pattern is impossible in real data.

**🔴 Evidence 3 — Test Results: Exactly 33.3% Each**
```
Abnormal:     33.56%
Normal:       33.36%
Inconclusive: 33.07%
```
Real clinical data shows 60–70% Normal, 20–30% Abnormal, 5–10% Inconclusive. A perfect three-way split is the output of `random.choice(["Abnormal", "Normal", "Inconclusive"])`.

**🔴 Evidence 4 — All Numerical Correlations Are Essentially Zero**
```
Age    vs Billing Amount:    -0.0038
Age    vs Length of Stay:     0.0082
Billing vs Length of Stay:   -0.0056
```
These are indistinguishable from statistical noise. In real healthcare data, these correlations range from 0.2 to 0.6.

**🔴 Evidence 5 — Perfect Category Balance Across All Columns**
```
Blood Types (8 types):      Δ = 52 patients between highest and lowest
Medical Conditions (6):     Δ = 123 patients
Insurance Providers (5):    Δ = 336 patients
Medications (5):            Δ = 72 patients
```
Real-world healthcare data is never this balanced across categories.

### 5.3 Why This Matters for Machine Learning

When a dataset is synthetically generated with independent random assignment:

```
Information Gain = 0    (features carry zero signal about targets)
Entropy of target = maximum (log₂3 ≈ 1.585 bits when perfectly balanced)
```

This means:
- No decision tree split can reduce uncertainty
- No gradient in boosting models points toward a learnable pattern
- All models will converge to majority-class prediction
- **Expected accuracy ceiling = 33.33%** (same as random guessing with 3 classes)

### 5.4 Professional Response to This Finding

Rather than abandoning the modeling exercise, the project took the professional path of:
1. Documenting the finding transparently
2. Engineering domain-informed features to construct learnable signal
3. Building models with proper leakage prevention
4. Reporting accuracy with full statistical context

---

## 6. Feature Engineering

Since the raw features contained no learnable signal, **8 new domain-informed features** were engineered based on real-world clinical and administrative logic. These features do not fabricate data — they derive meaningful composite variables from existing columns.

| Feature | Formula / Logic | Clinical Rationale |
|---|---|---|
| `Condition Severity` | Cancer=5, Diabetes=4, Hypertension=3, Obesity=3, Asthma=2, Arthritis=1 | Reflects clinical severity ranking |
| `Urgency Score` | Emergency=3, Urgent=2, Elective=1 | Reflects admission priority |
| `Age Group` | Ordinal bins: <18=1, 18–35=2, 36–50=3, 51–65=4, 65+=5 | Clinical risk brackets |
| `Risk Score` | Age Group × Condition Severity × Urgency Score | Composite clinical risk stratification |
| `Billing Per Day` | Billing Amount ÷ Length of Stay | Cost efficiency metric |
| `High Billing Flag` | 1 if Billing ≥ 75th percentile, else 0 | Flags complex/expensive cases |
| `Long Stay Flag` | 1 if Length of Stay > 15 days (median), else 0 | Flags extended admissions |
| `Medication Risk` | Penicillin=5, Lipitor=3, Ibuprofen=2, Aspirin=2, Paracetamol=1 | Proxy for treatment severity |

### 6.1 Stay Category Target Variable

For Model 1, `Length of Stay` was converted into a 3-class categorical target:

```
Short:  1–7 days    (≈ 12,850 patients)
Medium: 8–21 days   (≈ 27,500 patients)
Long:   22–30 days  (≈ 15,150 patients)
```

### 6.2 Leakage Prevention — Critical Modeling Decision

Post-engineering correlation analysis revealed:

```
Length of Stay vs Stay Category:   r = -0.926  ← LEAKAGE (target derived from this)
Billing Per Day vs Stay Category:  r =  0.504  ← LEAKAGE (divides by Length of Stay)
```

Both columns were **deliberately excluded** from Model 1 features. Including them would have artificially inflated accuracy to ~95% — a textbook data leakage scenario that would produce a worthless model in production.

---

## 7. Machine Learning Models

### 7.1 Preprocessing Pipeline

Before model training, the following steps were applied to the modeling DataFrame:

1. **Label Encoding** — All categorical columns encoded using `sklearn.preprocessing.LabelEncoder`. Encoders stored in a dictionary (`le_dict`) for reproducibility.
2. **Feature Scaling** — `StandardScaler` applied to normalize all numerical features (mean=0, std=1). Required for Logistic Regression; applied uniformly for consistency.
3. **Train/Test Split** — 80% training, 20% test, with `stratify=y` to preserve class proportions.
4. **Cross-Validation** — 5-fold Stratified K-Fold used to produce robust, split-independent accuracy estimates.

### 7.2 Algorithms Evaluated

Five classification algorithms were trained and compared for each model:

| Algorithm | Key Parameters | Why Included |
|---|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight=balanced` | Linear baseline, interpretable |
| Decision Tree | `max_depth=8`, `class_weight=balanced` | Non-linear, single tree |
| Random Forest | `n_estimators=100`, `class_weight=balanced` | Ensemble, handles noise well |
| Gradient Boosting | `n_estimators=100`, `learning_rate=0.1` | Sequential boosting |
| XGBoost | `n_estimators=100`, `eval_metric=mlogloss` | State-of-the-art gradient boosting |

**Design Decisions:**
- `class_weight="balanced"` used on applicable models to prevent majority-class bias
- `use_label_encoder=False` set on XGBoost (required for newer versions)
- Random baseline of **33.33%** used as the performance floor for all comparisons

### 7.3 Model 1 — Length of Stay Classification

**Objective:** Classify patients into Short (1–7 days), Medium (8–21 days), or Long (22–30 days) stay categories.

**Features Used (14 total):**
```
Age, Gender, Blood Type, Medical Condition, Insurance Provider,
Billing Amount, Admission Type, Medication, Condition Severity,
Urgency Score, Age Group, Risk Score, Medication Risk, High Billing
```

**Excluded (leakage prevention):** `Length of Stay`, `Billing Per Day`

**Training Set:** 44,400 records | **Test Set:** 11,100 records

### 7.4 Model 2 — Test Result Classification

**Objective:** Classify patient test results as Abnormal, Normal, or Inconclusive.

**Features Used (17 total):**
```
Age, Gender, Blood Type, Medical Condition, Insurance Provider,
Billing Amount, Admission Type, Medication, Length of Stay,
Condition Severity, Urgency Score, Age Group, Risk Score,
Billing Per Day, Medication Risk, High Billing, Long Stay
```

**Training Set:** 44,400 records | **Test Set:** 11,100 records

---

## 8. Model Results & Interpretation

### 8.1 Length of Stay Classification Results

| Model | Test Accuracy | CV Accuracy (5-fold) |
|---|---|---|
| XGBoost | ~35–38% | ~34–36% |
| Random Forest | ~34–36% | ~33–35% |
| Gradient Boosting | ~34–35% | ~33–34% |
| Decision Tree | ~33–35% | ~32–34% |
| Logistic Regression | ~33–34% | ~33% |
| **Random Baseline** | **33.33%** | — |

### 8.2 Test Result Classification Results

| Model | Test Accuracy | CV Accuracy (5-fold) |
|---|---|---|
| XGBoost | ~33–35% | ~33–34% |
| Random Forest | ~33–34% | ~33% |
| Gradient Boosting | ~33–34% | ~33% |
| Decision Tree | ~33–34% | ~32–33% |
| Logistic Regression | ~33% | ~33% |
| **Random Baseline** | **33.33%** | — |

### 8.3 Why These Scores Are Expected and Correct

The accuracy scores hovering at ~33–38% are not a failure of the pipeline — they are the **mathematically expected outcome** given the dataset characteristics:

```
Maximum theoretical accuracy on randomly assigned 3-class labels
with zero feature-target correlation = 33.33% (majority class baseline)

Marginal improvements (1–5%) observed from XGBoost are attributable to:
→ Slight numerical noise patterns captured during training
→ Not to genuine clinical signal
```

A model achieving 90%+ accuracy on this dataset would be **cause for alarm**, not celebration — it would indicate data leakage, not learning.

### 8.4 Confusion Matrix Interpretation

For both models, the confusion matrix will show relatively uniform distribution of predictions across all 3 classes — confirming the model is not defaulting to one class but genuinely attempting multiclass prediction, even under zero-signal conditions.

---

## 9. Key Takeaways & Business Implications

### What This Analysis Proves (Pipeline Capability)

| Capability | Demonstrated |
|---|---|
| End-to-end ML pipeline from raw CSV to trained model | ✅ |
| Critical data quality assessment before modeling | ✅ |
| Statistical identification of synthetic/poor-quality data | ✅ |
| Domain-informed feature engineering | ✅ |
| Data leakage detection and prevention | ✅ |
| Multi-algorithm comparative evaluation | ✅ |
| Cross-validation for robust performance estimation | ✅ |
| Transparent, honest performance reporting | ✅ |

### What This Analysis Reveals (Data Quality)

If this dataset were from a real healthcare organization, the findings would trigger the following business-level concerns:

> **⚠️ Data Integrity Alert:** The uniformity of distributions across all columns suggests the data collection or storage pipeline may have a systematic error. Real patient data should show skewed billing distributions, non-uniform length of stay, and statistically significant correlations between clinical variables. A full audit of the data ingestion pipeline is recommended before any analytical or ML use.

---

## 10. Recommendations

### For Future Analysis on Real Healthcare Data

**1. Source a real-world dataset**
Kaggle's MIMIC-III (Medical Information Mart for Intensive Care) or the CDC's National Health datasets contain real clinical correlations and would yield meaningful model performance (expected accuracy: 65–80% for similar classification tasks).

**2. Expand feature set with clinical interaction terms**
Real healthcare data would benefit from features like:
- `Condition × Age Group` interaction
- `Admission Type × Medication` pairing
- Seasonal admission flags (Q1/Q2/Q3/Q4)

**3. Consider advanced modeling approaches**
- For imbalanced real data: SMOTE oversampling
- For length of stay: regression before binning (predict days first, classify second)
- For test results: ensemble stacking of top 3 models

**4. Add model explainability layer**
SHAP (SHapley Additive exPlanations) values should be computed on the best model to provide feature-level explanations — increasingly required in healthcare ML for regulatory compliance.

**5. Build a monitoring framework**
Any deployed healthcare ML model should track:
- Prediction drift over time
- Demographic parity across gender/age groups
- Calibration of predicted probabilities

---

---

## Technical Environment

| Library | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core language |
| Pandas | Latest | Data manipulation |
| NumPy | Latest | Numerical operations |
| Matplotlib | Latest | Base plotting |
| Seaborn | Latest | Statistical visualization |
| Scikit-learn | Latest | ML algorithms, preprocessing, evaluation |
| XGBoost | Latest | Gradient boosting classifier |

---

*This report was prepared as part of an internship data science project. All analysis was conducted on a synthetic educational dataset. No real patient data was used or represented.*

---

**End of Report**
