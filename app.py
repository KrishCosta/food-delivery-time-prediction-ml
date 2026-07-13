from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Food Delivery Time Prediction",
    layout="centered"
)

PROJECT_PATH = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_PATH / "models" / "food_delivery_model.pkl"


NUMERICAL_FEATURES = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs"
]

CATEGORICAL_FEATURES = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type"
]

MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


@st.cache_resource
def load_model():
    """Load and cache the trained machine-learning pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def get_category_options(model):
    """Read the fitted categorical values from the model pipeline."""
    preprocessor = model.named_steps["preprocessor"]

    categorical_pipeline = (preprocessor.named_transformers_["categorical"])

    encoder = categorical_pipeline.named_steps["encoder"]

    return {
        feature_name: [
            str(category)
            for category in categories
        ]
        for feature_name, categories in zip(
            CATEGORICAL_FEATURES,
            encoder.categories_
        )
    }

try:
    model = load_model()
    category_options = get_category_options(model)

except Exception as error:
    st.error(
        "The trained model could not be loaded. "
        "Check that models/food_delivery_model.pkl exists "
        "and that the required package versions are installed."
    )
    st.exception(error)
    st.stop()

st.title("Food Delivery Time Prediction")

st.write(
    "Enter the order and delivery information below to estimate the expected delivery time."
)

model_name = model.named_steps["model"].__class__.__name__

st.caption(f"Prediction model: {model_name}")

with st.form("food_delivery_prediction_form"):
    st.subheader("Delivery information")
    left_column, right_column = st.columns(2)

    with left_column:
        distance_km = st.number_input(
            "Delivery distance (km)",
            min_value=0.0,
            value=5.0,
            step=0.1,
            format="%.1f"
        )
        preparation_time = st.number_input(
            "Preparation time (minutes)",
            min_value=0,
            value=15,
            step=1,
            format="%d"
        )
        courier_experience = st.number_input(
            "Courier experience (years)",
            min_value=0,
            value=3,
            step=1,
            format="%d"
        )
        weather = st.selectbox(
            "Weather condition",
            options=category_options["Weather"]
        )

    with right_column:
        traffic_level = st.selectbox(
            "Traffic level",
            options=category_options["Traffic_Level"]
        )
        time_of_day = st.selectbox(
            "Time of day",
            options=category_options["Time_of_Day"]
        )
        vehicle_type = st.selectbox(
            "Vehicle type",
            options=category_options["Vehicle_Type"]
        )
    submitted = st.form_submit_button(
        "Predict delivery time",
        use_container_width=True
    )

if submitted:

    input_data = pd.DataFrame(
        [
            {
                "Distance_km": distance_km,
                "Preparation_Time_min": preparation_time,
                "Courier_Experience_yrs": courier_experience,
                "Weather": weather,
                "Traffic_Level": traffic_level,
                "Time_of_Day": time_of_day,
                "Vehicle_Type": vehicle_type
            }
        ],
        columns=MODEL_FEATURES
    )

    try:
        predicted_minutes = float(model.predict(input_data)[0])
        st.success("Prediction completed successfully.")

        st.metric(
            "Estimated delivery time",
            f"{predicted_minutes:.2f} minutes"
        )

        if predicted_minutes < 0:
            st.warning(
                "The model returned a negative estimate. "
                "Review whether the entered values are within "
                "the range represented by the training data."
            )

        with st.expander("View submitted information"):
            st.dataframe(
                input_data,
                hide_index=True
            )

    except Exception as error:
        st.error( "The prediction could not be completed.")
        st.exception(error)

st.divider()

st.caption(
    "This estimate is produced by a regression model trained on historical food-delivery data."
)