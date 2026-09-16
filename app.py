import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾")
st.title("🌾 Crop Yield Prediction")
st.write("Predict crop yield based on crop, season, state, and inputs.")

# ---------- Load model + column list + category options ----------
model = joblib.load("decision_tree_model.pkl")
model_columns = joblib.load("model_columns.pkl")
categories = joblib.load("categories.pkl")  # {"Crop": [...], "Season": [...], "State": [...]}

# Fail fast with a clear message instead of a cryptic sklearn error
# if the model and column list ever get out of sync again.
n_expected = getattr(model, "n_features_in_", None)
if n_expected is not None and n_expected != len(model_columns):
    st.error(
        f"Model/columns mismatch: the model expects {n_expected} features, "
        f"but model_columns.pkl has {len(model_columns)}. "
        "Re-run train_model.py so both files come from the same training run."
    )
    st.stop()


def predict_yield(crop, season, state, crop_year, rainfall, area, fertilizer, pesticide):
    """Build one model-ready row and return the predicted yield. Reused
    for both the manual form below and the built-in example cases."""
    row = {
        "Crop_Year": crop_year,
        "Annual_Rainfall": rainfall,
        "Area_log": np.log1p(area),
        "Fertilizer_log": np.log1p(fertilizer),
        "Pesticide_log": np.log1p(pesticide),
    }
    for col in model_columns:
        if col not in row:
            row[col] = 0

    crop_col   = f"Crop_{crop}"
    season_col = f"Season_{season}"
    state_col  = f"State_{state}"
    if crop_col in row:
        row[crop_col] = 1
    if season_col in row:
        row[season_col] = 1
    if state_col in row:
        row[state_col] = 1

    X = pd.DataFrame([row])[model_columns]
    pred_log = model.predict(X)[0]
    return float(np.expm1(pred_log))


# ---------- Keep a running log of every prediction this session ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Sample test cases (for verification) ----------
# 3 fixed example inputs, predicted live using the model that's already
# loaded above. No separate file, no retraining needed — whatever model
# is currently deployed is exactly what generates these numbers, so
# they're always guaranteed to match what the app would give you.
SAMPLE_INPUTS = [
    {
        "Crop": categories["Crop"][0],
        "Season": categories["Season"][0],
        "State": categories["State"][0],
        "Crop_Year": 2010,
        "Annual_Rainfall": 1000.0,
        "Area": 5000.0,
        "Fertilizer": 500000.0,
        "Pesticide": 2000.0,
    },
    {
        "Crop": categories["Crop"][len(categories["Crop"]) // 2],
        "Season": categories["Season"][len(categories["Season"]) // 2],
        "State": categories["State"][len(categories["State"]) // 2],
        "Crop_Year": 2015,
        "Annual_Rainfall": 1500.0,
        "Area": 20000.0,
        "Fertilizer": 2000000.0,
        "Pesticide": 8000.0,
    },
    {
        "Crop": categories["Crop"][-1],
        "Season": categories["Season"][-1],
        "State": categories["State"][-1],
        "Crop_Year": 2020,
        "Annual_Rainfall": 2200.0,
        "Area": 50000.0,
        "Fertilizer": 5000000.0,
        "Pesticide": 20000.0,
    },
]

st.subheader("✅ Sample Test Cases (for verification)")
st.caption(
    "Fixed example inputs with the model's predicted output shown below each one. "
    "Click 'Load Example' to fill the form with it, then click Predict Yield — "
    "the result should match the number shown here."
)

ex_cols = st.columns(len(SAMPLE_INPUTS))
for i, (col, ex) in enumerate(zip(ex_cols, SAMPLE_INPUTS)):
    pred = predict_yield(
        ex["Crop"], ex["Season"], ex["State"], ex["Crop_Year"],
        ex["Annual_Rainfall"], ex["Area"], ex["Fertilizer"], ex["Pesticide"],
    )
    with col:
        st.markdown(f"**Example {i + 1}**")
        st.write(f"Crop: {ex['Crop']}")
        st.write(f"Season: {ex['Season']}")
        st.write(f"State: {ex['State']}")
        st.write(f"Year: {ex['Crop_Year']}")
        st.write(f"Rainfall: {ex['Annual_Rainfall']}")
        st.write(f"Area: {ex['Area']}")
        st.write(f"Fertilizer: {ex['Fertilizer']}")
        st.write(f"Pesticide: {ex['Pesticide']}")
        st.write(f"**Expected Output: {pred:.3f}**")

        def _load(ex=ex):
            st.session_state["in_crop"] = ex["Crop"]
            st.session_state["in_season"] = ex["Season"]
            st.session_state["in_state"] = ex["State"]
            st.session_state["in_crop_year"] = ex["Crop_Year"]
            st.session_state["in_rainfall"] = ex["Annual_Rainfall"]
            st.session_state["in_area"] = ex["Area"]
            st.session_state["in_fertilizer"] = ex["Fertilizer"]
            st.session_state["in_pesticide"] = ex["Pesticide"]

        st.button(f"Load Example {i + 1}", key=f"load_ex_{i}", on_click=_load)

st.divider()

# Defaults for the input widgets below — only applied the first time,
# or overwritten instantly when a "Load Example" button is clicked above.
st.session_state.setdefault("in_crop", categories["Crop"][0])
st.session_state.setdefault("in_season", categories["Season"][0])
st.session_state.setdefault("in_state", categories["State"][0])
st.session_state.setdefault("in_crop_year", 2015)
st.session_state.setdefault("in_rainfall", 1200.0)
st.session_state.setdefault("in_area", 10000.0)
st.session_state.setdefault("in_fertilizer", 1e6)
st.session_state.setdefault("in_pesticide", 5000.0)

# ---------- Input UI ----------
crop   = st.selectbox("Crop", categories["Crop"], key="in_crop")
season = st.selectbox("Season", categories["Season"], key="in_season")
state  = st.selectbox("State", categories["State"], key="in_state")

crop_year  = st.number_input("Crop Year", 1997, 2020, key="in_crop_year")
rainfall   = st.number_input("Annual Rainfall (mm)", 0.0, 7000.0, key="in_rainfall")
area       = st.number_input("Area (raw)", 0.0, 1e8, key="in_area")
fertilizer = st.number_input("Fertilizer (raw)", 0.0, 1e10, key="in_fertilizer")
pesticide  = st.number_input("Pesticide (raw)", 0.0, 1e8, key="in_pesticide")

# Let the user double check exactly what will be sent to the model
# before they hit predict.
with st.expander("Review your inputs"):
    st.table(pd.DataFrame([{
        "Crop": crop,
        "Season": season,
        "State": state,
        "Crop Year": crop_year,
        "Annual Rainfall (mm)": rainfall,
        "Area (raw)": area,
        "Fertilizer (raw)": fertilizer,
        "Pesticide (raw)": pesticide,
    }]))

# ---------- Predict ----------
if st.button("Predict Yield"):
    try:
        pred_yield = predict_yield(crop, season, state, crop_year, rainfall, area, fertilizer, pesticide)
        st.success(f"🌱 Predicted Yield: **{pred_yield:.3f}**")

        st.session_state.history.insert(0, {
            "Crop": crop,
            "Season": season,
            "State": state,
            "Crop Year": crop_year,
            "Rainfall (mm)": rainfall,
            "Area": area,
            "Fertilizer": fertilizer,
            "Pesticide": pesticide,
            "Predicted Yield": round(pred_yield, 3),
        })
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------- Prediction history ----------
if st.session_state.history:
    st.subheader("📋 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download history as CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv",
    )

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()