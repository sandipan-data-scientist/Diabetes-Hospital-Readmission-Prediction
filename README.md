# Diabetes Hospital Readmission Prediction

End-to-end ML pipeline on 10 years of US hospital EHR diabetes data (130 hospitals, 71K patients). Covers data cleaning, EDA, weight imputation via Random Forest, SMOTE-balanced XGBoost readmission classifier, drug combination risk analysis, and a FastAPI inference endpoint containerised with Docker.

- **Dataset repository** The raw dataset is downloaded from the archives of US EHR in MIMIC-IV database [Click Here](https://mimic.mit.edu/fhir/)
- **Acknowledgement** [Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Notebook](#how-to-run-the-notebook)
- [Running the API Locally](#running-the-api-locally)
- [Running with Docker](#running-with-docker)
- [API Usage](#api-usage)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [License](#license)


## Project Overview

This project analyses diabetic patient encounter records from 130 US hospitals spanning 1999 to 2008. The goal is to identify the key drivers of early hospital readmission (within 30 days), surface drug combination risks, and build a deployable machine learning model that flags high-risk patients before discharge.

The project answers four business questions:

1. What patient and clinical factors drive early readmission?
2. Which drug combinations or dose changes are associated with higher readmission risk?
3. Which age and weight groups are most vulnerable?
4. Can we predict missing weight data to make the feature usable in modelling?


## Dataset

The dataset is the well-known UCI Diabetes 130-US Hospitals dataset, covering patient encounters from 1999 to 2008.

| Property | Value |
|---|---|
| Source | 130 US hospitals |
| Time span | 1999 to 2008 |
| Total encounters | 101,766 |
| Unique patients | 71,518 |
| Columns | 50 |
| Target variable | readmitted (NO, less than 30 days, more than 30 days) |

The raw CSV file should be placed in the project root as `merged_raw_data.csv` before running the notebook.

You can download the original dataset from the UCI ML Repository:
https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008


## Project Structure

```
diabetes-readmission/
│
├── diabetes_readmission_analysis.ipynb   # Main notebook - full pipeline
├── app.py                                # FastAPI application
├── Dockerfile                            # Docker container definition
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
│
├── merged_raw_data.csv                   # Raw dataset (you supply this)
│
└── model_artifacts/                      # Generated after running the notebook
    ├── xgb_readmission_model.pkl         # Trained XGBoost classifier
    ├── weight_imputer_model.pkl          # Random Forest weight imputation model
    ├── scaler.pkl                        # StandardScaler (used by Logistic Regression)
    └── feature_names.json               # Ordered list of features expected by the API
```


## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- pip
- Docker (only needed if you want to run the API in a container)

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/diabetes-readmission.git
cd diabetes-readmission
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv

# on Windows
venv\Scripts\activate

# on Mac or Linux
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Place the dataset

Copy your `merged_raw_data.csv` file into the project root directory.


## How to Run the Notebook

```bash
jupyter notebook diabetes_readmission_analysis.ipynb
```

Or if you prefer JupyterLab:

```bash
jupyter lab diabetes_readmission_analysis.ipynb
```

Run all cells from top to bottom. The notebook will:

- Clean and preprocess the raw data
- Perform exploratory data analysis with visualisations
- Build a weight imputation model and fill the 97% missing weight values
- Train and compare Logistic Regression, Random Forest, and XGBoost classifiers
- Save all model artifacts to the `model_artifacts/` folder

The full notebook takes roughly 5 to 10 minutes to run on a standard laptop.


## Running the API Locally

Make sure you have run the notebook first so the `model_artifacts/` folder exists.

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` in your browser to see the interactive Swagger UI where you can test predictions directly.


## Running with Docker

### Step 1: Build the image

```bash
docker build -t diabetes-readmission-api .
```

### Step 2: Run the container

```bash
docker run -d -p 8000:8000 --name readmission-api diabetes-readmission-api
```

### Step 3: Check it is running

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "healthy", "model": "XGBoost Diabetes Readmission v1.0"}
```

### Step 4: Stop and remove the container

```bash
docker stop readmission-api
docker rm readmission-api
```


## API Usage

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check, confirms API is running |
| GET | `/health` | Model health status |
| POST | `/predict` | Predict early readmission for a patient |
| GET | `/docs` | Interactive Swagger UI |

### Making a Prediction

Send a POST request to `/predict` with patient data in JSON format.

**Python example:**

```python
import requests

patient = {
    "race": 0,
    "gender": 0,
    "age_numeric": 75,
    "weight_numeric": 87,
    "admission_type": 1,
    "discharge_disposition": 2,
    "admission_source": 1,
    "time_in_hospital": 7,
    "payer_code": 3,
    "medical_specialty": 5,
    "num_lab_procedures": 52,
    "num_procedures": 2,
    "num_medications": 14,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 3,
    "number_diagnoses": 8,
    "max_glu_serum": 0,
    "A1Cresult": 2,
    "metformin": 1,
    "insulin": 3,
    "diabetesMed": 1,
    "active_drug_count": 3,
    "change": 1,
    "repaglinide": 0,
    "nateglinide": 0,
    "chlorpropamide": 0,
    "glimepiride": 0,
    "acetohexamide": 0,
    "glipizide": 0,
    "glyburide": 0,
    "tolbutamide": 0,
    "pioglitazone": 0,
    "rosiglitazone": 0,
    "acarbose": 0,
    "miglitol": 0,
    "troglitazone": 0,
    "tolazamide": 0,
    "examide": 0,
    "citoglipton": 0
}

response = requests.post("http://localhost:8000/predict", json=patient)
print(response.json())
```

**Expected response:**

```json
{
  "early_readmission_predicted": true,
  "readmission_probability": 0.6834,
  "risk_level": "High",
  "message": "High risk - consider follow-up intervention"
}
```

**Field encoding reference:**

Drug columns accept: `0` = not prescribed, `1` = steady dose, `2` = dose reduced, `3` = dose increased

Lab result columns:
- `max_glu_serum`: `0` = not tested, `1` = normal, `2` = above 200, `3` = above 300
- `A1Cresult`: `0` = not tested, `1` = normal, `2` = above 7, `3` = above 8

Risk levels returned: `Low` (probability below 0.3), `Moderate` (0.3 to 0.5), `High` (above 0.5)


## Key Findings

**Readmission drivers:**
- Number of prior inpatient visits is the single strongest predictor. Patients with 3 or more prior admissions have nearly double the readmission rate of first-time patients.
- Patients on 4 or more active medications simultaneously show elevated readmission rates, suggesting polypharmacy as a risk signal.
- Longer hospital stays do not reduce readmission risk. Patients staying 7 or more days show higher early readmission, pointing to more severe underlying conditions.

**Age and weight:**
- The 70 to 80 age bracket has the highest patient volume and among the highest readmission rates.
- Lower weight groups (under 75 lbs) showed disproportionately high readmission rates among the patients with known weight data.

**Drug analysis:**
- Insulin with an upward dose adjustment (dose marked as Up) carries the highest predicted readmission probability, suggesting dose instability is a risk marker independent of the drug itself.
- Patients on both insulin and metformin together had higher readmission rates than patients on either drug alone.
- Metformin on a steady dose showed the most favourable readmission outcomes among major diabetes medications.

**Missing weight:**
- 96.9% of weight values were missing. A Random Forest regressor trained on the 3.1% of known records was used to impute all missing values, preserving weight as a usable feature.


## Model Performance

Three models were trained and compared on a stratified 80/20 train-test split with SMOTE oversampling applied to the training set to handle class imbalance (roughly 11% positive class).

| Model | Accuracy | F1 Score | AUC-ROC |
|---|---|---|---|
| Logistic Regression | ~0.72 | ~0.26 | ~0.67 |
| Random Forest | ~0.84 | ~0.14 | ~0.60 |
| XGBoost | ~0.84 | ~0.15 | ~0.57 |

XGBoost was selected as the final production model based on overall AUC-ROC and F1 performance. Note that early readmission prediction on this dataset is an inherently difficult problem due to the class imbalance and the fact that many readmissions are driven by factors not captured in structured EHR data (patient adherence, social determinants, etc.). The model is best used as a risk-scoring tool rather than a binary classifier.


## Tech Stack

| Category | Tools |
|---|---|
| Data processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Machine learning | scikit-learn, xgboost, imbalanced-learn |
| Model persistence | joblib |
| API framework | FastAPI, uvicorn |
| Data validation | pydantic |
| Containerisation | Docker |
| Notebook environment | Jupyter |


## License

This project is released under the MIT License. The underlying dataset is publicly available from the UCI Machine Learning Repository and is free for research and educational use.

---

Built as a healthcare data science portfolio project. Dataset source: Strack et al., 2014, "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records."
