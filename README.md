# AI-Based Food Waste Prediction & Reduction System

Machine Learning PBL project. A regression model estimates the total food waste (tons) recorded for a country, year and food category, and a Streamlit app adds a reduction-scenario calculator and a ranking of food categories to target first.

**Team:** Om Gorde (2124UMLM1100), Aayush Surwase (2124UMLM1058), Pranav Dange (2124UMLM1081), Dewang Gorhe (2124UMLM1007)

## Dataset
Kaggle, *Global Food Wastage Dataset (2018-2024)* (5,000 records, 20 countries, 8 food categories). It is a secondary dataset with undocumented collection method, so results describe patterns in this dataset and are not official statistics.

## Method
- Target: `Total Waste (Tons)`
- Features: country, food category, year, population (million), household waste (%)
- Excluded to avoid leakage: Economic Loss, Avg Waste per Capita
- Models: Dummy median, Linear Regression, Decision Tree, Random Forest
- 80:20 split (`random_state=42`), 5-fold CV on training data, model chosen by CV MAE

## How to run
1. Download the CSV from Kaggle and save it as `data/global_food_wastage_dataset.csv`.
2. Install packages: `python -m pip install -r requirements.txt`
3. Open `notebooks/food_waste_prediction.ipynb` and run all cells in order.
4. Confirm `best_model.joblib`, `country_reference.csv` and `reduction_reference.json` were created in the project root.
5. Run: `streamlit run app.py`

## Results
_Fill in after running the notebook (CV MAE, test MAE, RMSE, R2 per model, and the selected model)._

## Limitations
Estimates the dataset's recorded waste for 2018-2024 only; not a forecast, live figure or guaranteed saving. The loss-avoided number in the app multiplies avoided tons by the dataset's median loss per ton.
