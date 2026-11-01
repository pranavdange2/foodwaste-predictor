import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Food Waste Predictor")
st.title("AI-Based Food Waste Prediction & Reduction System")
st.write("Estimate recorded food waste for a country, year and food category, "
         "then test a reduction scenario.")

FILES = ["best_model.joblib", "country_reference.csv",
         "reduction_reference.json"]
if not all(Path(f).exists() for f in FILES):
    st.error("Setup incomplete: run the notebook first so that "
             + ", ".join(FILES) + " exist in this folder.")
    st.stop()

model = joblib.load("best_model.joblib")
ref = pd.read_csv("country_reference.csv").set_index("country")
loss_per_ton = json.loads(
    Path("reduction_reference.json").read_text())["median_loss_usd_per_ton"]

encoder = (model.named_steps["preprocess"]
           .named_transformers_["cat"].named_steps["onehot"])
countries = list(encoder.categories_[0])
categories = list(encoder.categories_[1])

country = st.selectbox("Country", countries)
food_category = st.selectbox("Food category", categories)
year = st.number_input("Year", min_value=2018, max_value=2024,
                       value=2024, step=1)
population = st.number_input(
    "Population (million)", min_value=0.1, step=1.0,
    value=float(ref.loc[country, "population_million"]),
    key=f"pop_{country}")
household = st.slider(
    "Household share of waste (%)", 0.0, 100.0,
    float(ref.loc[country, "household_waste_pct"]), key=f"hh_{country}")
reduction = st.slider("Reduction target for the scenario (%)", 0, 50, 10)

def predict_tons(category):
    row = pd.DataFrame([{
        "year": int(year),
        "population_million": float(population),
        "household_waste_pct": float(household),
        "country": country,
        "food_category": category,
    }])
    return max(0.0, float(model.predict(row)[0]))

if st.button("Estimate waste"):
    estimate = predict_tons(food_category)
    avoided = estimate * reduction / 100
    saved_musd = avoided * loss_per_ton / 1e6
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated waste (tons)", f"{estimate:,.0f}")
    c2.metric(f"Avoided at {reduction}% cut (tons)", f"{avoided:,.0f}")
    c3.metric("Estimated loss avoided (million USD)", f"{saved_musd:,.2f}")

    ranking = pd.DataFrame({
        "food_category": categories,
        "estimated_tons": [predict_tons(c) for c in categories],
    }).sort_values("estimated_tons", ascending=False)
    st.subheader("Where to focus reduction effort")
    st.bar_chart(ranking.set_index("food_category"))
    st.dataframe(ranking, hide_index=True)
    st.caption("Educational estimate from a 2018-2024 secondary dataset; "
               "not an official statistic or a guarantee. The loss figure "
               "uses the dataset's median loss per ton.")
