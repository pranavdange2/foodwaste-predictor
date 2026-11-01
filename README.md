# AI-Based Food Waste Prediction & Reduction System

A machine learning project that estimates how much food waste (in tons) a country records for a given year and food category, and helps plan where to cut it. A Streamlit web app lets users pick a country and food category, get a predicted waste figure, test a reduction target, and see which food categories contribute the most waste.

Built as a Machine Learning PBL project.

## Features

- **Waste prediction:** regression model trained on multi-country food waste records (2018-2024).
- **Reduction scenario:** choose a 0-50% reduction target and see the tons and estimated economic loss avoided.
- **Category ranking:** compares predicted waste across all food categories for the selected country, so high-waste categories can be targeted first.
- **Reproducible pipeline:** cleaning, encoding and modelling are wrapped in a scikit-learn pipeline and saved with joblib.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit, joblib, Jupyter

## Dataset

[Global Food Wastage Dataset (2018-2024)](https://www.kaggle.com/datasets?search=global+food+wastage+dataset+2018-2024) from Kaggle: 5,000 records covering 20 countries and 8 food categories.

| Column | Role |
|---|---|
| Total Waste (Tons) | Target |
| Country, Food Category | Input (categorical) |
| Year, Population (Million), Household Waste (%) | Input (numeric) |
| Economic Loss (Million $), Avg Waste per Capita (Kg) | Excluded from the model to avoid target leakage; used for analysis and the reduction scenario |

The dataset is a secondary source with an undocumented collection method, so results describe patterns in this data rather than official statistics.

## Project structure

```
Food-Waste-Prediction-Reduction/
├── data/                         # place the Kaggle CSV here
├── notebooks/
│   └── food_waste_prediction.ipynb   # cleaning, EDA, training, evaluation
├── app.py                        # Streamlit application
├── best_model.joblib             # saved pipeline (generated)
├── country_reference.csv         # app defaults per country (generated)
├── reduction_reference.json      # loss-per-ton constant (generated)
├── requirements.txt
└── README.md
```

## Getting started

```bash
# 1. Clone the repository and enter the folder
git clone <your-repository-url>
cd Food-Waste-Prediction-Reduction

# 2. (Optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
python -m pip install -r requirements.txt
```

Download the dataset from Kaggle and save it as `data/global_food_wastage_dataset.csv`.

## Usage

**1. Train the model.** Open `notebooks/food_waste_prediction.ipynb` and run all cells in order. The last cell creates `best_model.joblib`, `country_reference.csv` and `reduction_reference.json` in the project root.

**2. Launch the app.**

```bash
streamlit run app.py
```

Choose a country, food category and year, set a reduction target, and click **Estimate waste**.

## Methodology

1. **Cleaning:** standardise column names, convert types, remove duplicates and invalid rows.
2. **EDA:** distribution of waste, comparison by category, country and year, correlation analysis.
3. **Leakage check:** economic loss and per-capita waste are excluded because they are closely tied to the target.
4. **Modelling:** Dummy baseline, Linear Regression, Decision Tree and Random Forest inside a pipeline (median imputation for numeric fields, most-frequent imputation and one-hot encoding for categorical fields).
5. **Evaluation:** 80:20 train-test split (`random_state=42`), 5-fold cross-validation on the training set for model selection, final MAE, RMSE and R² on the held-out test set.

## Results

| Model | CV MAE (tons) | Test MAE (tons) | Test RMSE (tons) | Test R² |
|---|---|---|---|---|
| Dummy median | - | - | - | - |
| Linear Regression | - | - | - | - |
| Decision Tree | - | - | - | - |
| Random Forest | - | - | - | - |

Selected model: -

## Limitations

- Predictions cover the dataset's 2018-2024 range only; the system is not a forecast of future years.
- The estimate is of the dataset's recorded waste, not an official statistic or a live measurement.
- The loss-avoided figure is a scenario calculation (avoided tons x the dataset's median loss per ton), not a model prediction.
- Data is country-level, so it does not support restaurant- or household-level decisions.

## Team

| Name | Roll No. |
|---|---|
| Om Gorde | 2124UMLM1100 |
| Aayush Surwase | 2124UMLM1058 |
| Pranav Dange | 2124UMLM1081 |
| Dewang Gorhe | 2124UMLM1007 |
