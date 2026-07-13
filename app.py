"""
Hotel Reservation Cancellation Prediction - Streamlit App
-----------------------------------------------------------
A multi-tab deployment app for the hotel cancellation prediction model:

  - Predict:            single-booking form with risk tier + SHAP explanation
  - Batch Prediction:   upload a CSV of bookings, get predictions for all of them
  - Insights:           EDA-style charts on historical cancellation patterns
  - Model Performance:  confusion matrix, classification report, model comparison

Required files in the same folder as this script:
    - best_model_pipeline.pkl   (required)
    - column_info.pkl           (required)
    - dashboard_data.pkl        (optional - powers Insights & Model Performance tabs)
"""

import datetime
import io

import holidays as holidays_lib
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hotel Reservation Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# ---------------------------------------------------------------------
# Feature engineering (mirrors the notebook exactly)
# ---------------------------------------------------------------------
def compute_is_near_holiday(year, month, day, window_days=3):
    try:
        arrival = datetime.date(int(year), int(month), int(day))
    except ValueError:
        return 0
    pt_holidays = holidays_lib.Portugal(years=[int(year) - 1, int(year), int(year) + 1])
    for offset in range(-window_days, window_days + 1):
        check_date = arrival + datetime.timedelta(days=offset)
        if check_date in pt_holidays:
            return 1
    return 0


def engineer_features(input_df):
    """Adds every engineered feature the model expects, matching the training notebook."""
    df = input_df.copy()
    df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    df["total_guests"] = df["no_of_adults"] + df["no_of_children"]
    df["total_previous_bookings"] = (
        df["no_of_previous_cancellations"] + df["no_of_previous_bookings_not_canceled"]
    )
    df["prior_cancel_rate"] = (
        df["no_of_previous_cancellations"] / df["total_previous_bookings"].replace(0, np.nan)
    ).fillna(0)
    df["has_special_requests"] = (df["no_of_special_requests"] > 0).astype(int)
    df["is_near_holiday"] = df.apply(
        lambda r: compute_is_near_holiday(r["arrival_year"], r["arrival_month"], r["arrival_date"]),
        axis=1
    )
    return df


def get_risk_tier(proba):
    if proba < 0.30:
        return "Low", "🟢"
    elif proba < 0.60:
        return "Medium", "🟡"
    else:
        return "High", "🔴"


def validate_date(year, month, day):
    try:
        datetime.date(int(year), int(month), int(day))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("best_model_pipeline.pkl")
    column_info = joblib.load("column_info.pkl")
    try:
        dashboard_data = joblib.load("dashboard_data.pkl")
    except FileNotFoundError:
        dashboard_data = None
    return pipeline, column_info, dashboard_data


try:
    pipeline, column_info, dashboard_data = load_artifacts()
except FileNotFoundError:
    st.error(
        "Could not find `best_model_pipeline.pkl` and/or `column_info.pkl`. "
        "Run the notebook through the model-saving section first, then place "
        "both files in the same folder as this app.py."
    )
    st.stop()

cat_cols = column_info["cat_cols"]
cat_options = column_info["cat_options"]
best_model_name = column_info["best_model_name"]

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------
# SHAP explainability (tree models only)
# ---------------------------------------------------------------------
def get_shap_contributions(input_df, top_n=6):
    try:
        import shap
    except ImportError:
        return None

    model_step = pipeline.named_steps["model"]
    preprocessor_step = pipeline.named_steps["preprocessor"]

    if not hasattr(model_step, "feature_importances_"):
        return None  # SHAP TreeExplainer path below is for tree models

    transformed = preprocessor_step.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor_step.get_feature_names_out()

    explainer = shap.TreeExplainer(model_step)
    sv = explainer.shap_values(transformed)

    # Normalize output shape across SHAP versions
    if isinstance(sv, list):
        row_values = sv[1][0]  # class 1 = Cancelled
    elif sv.ndim == 3:
        row_values = sv[0, :, 1]
    else:
        row_values = sv[0]

    contrib_df = pd.DataFrame({"feature": feature_names, "shap_value": row_values})
    contrib_df["abs_value"] = contrib_df["shap_value"].abs()
    contrib_df = contrib_df.sort_values("abs_value", ascending=False).head(top_n)
    contrib_df["direction"] = np.where(contrib_df["shap_value"] > 0, "Increases risk", "Decreases risk")
    return contrib_df.drop(columns="abs_value")


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🏨 Hotel Reservation Cancellation Predictor")
st.caption(f"Model in use: **{best_model_name}**")

tab_predict, tab_batch, tab_insights, tab_performance = st.tabs(
    ["🔮 Predict", "📁 Batch Prediction", "📊 Insights", "📈 Model Performance"]
)

# =======================================================================
# TAB 1: PREDICT
# =======================================================================
with tab_predict:
    st.write(
        "Fill in the booking details below to estimate the probability that "
        "this reservation will be cancelled."
    )

    with st.form("booking_form"):
        st.subheader("Guest & Stay Details")
        col1, col2 = st.columns(2)

        with col1:
            no_of_adults = st.number_input("Number of adults", min_value=0, max_value=10, value=2)
            no_of_children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
            no_of_weekend_nights = st.number_input("Weekend nights", min_value=0, max_value=10, value=1)
            no_of_week_nights = st.number_input("Week nights", min_value=0, max_value=15, value=2)
            required_car_parking_space = st.selectbox("Requires car parking space?", ["No", "Yes"])
            no_of_special_requests = st.number_input("Number of special requests", min_value=0, max_value=5, value=0)

        with col2:
            lead_time = st.number_input(
                "Lead time (days between booking and arrival)",
                min_value=0, max_value=500, value=50
            )
            arrival_year = st.selectbox("Arrival year", [2017, 2018, 2026, 2027], index=1)
            arrival_month = st.selectbox("Arrival month", list(range(1, 13)), index=6)
            arrival_date = st.number_input("Arrival day of month", min_value=1, max_value=31, value=15)
            repeated_guest = st.selectbox("Repeated guest?", ["No", "Yes"])

        st.subheader("Booking Details")
        col3, col4 = st.columns(2)
        with col3:
            type_of_meal_plan = st.selectbox("Meal plan", cat_options.get("type_of_meal_plan", []))
            room_type_reserved = st.selectbox("Room type", cat_options.get("room_type_reserved", []))
        with col4:
            market_segment_type = st.selectbox("Market segment", cat_options.get("market_segment_type", []))
            avg_price_per_room = st.number_input(
                "Average price per room (euros)",
                min_value=0.0, max_value=600.0, value=100.0, step=1.0
            )

        st.subheader("Guest History")
        col5, col6 = st.columns(2)
        with col5:
            no_of_previous_cancellations = st.number_input(
                "Previous cancellations", min_value=0, max_value=50, value=0
            )
        with col6:
            no_of_previous_bookings_not_canceled = st.number_input(
                "Previous bookings NOT cancelled", min_value=0, max_value=100, value=0
            )

        submitted = st.form_submit_button("Predict Cancellation Risk", use_container_width=True)

    if submitted:
        if not validate_date(arrival_year, arrival_month, arrival_date):
            st.warning(
                f"⚠️ {arrival_year}-{arrival_month}-{arrival_date} is not a valid calendar date. "
                "Please double check the arrival month/day combination."
            )
        else:
            raw_input = {
                "no_of_adults": no_of_adults,
                "no_of_children": no_of_children,
                "no_of_weekend_nights": no_of_weekend_nights,
                "no_of_week_nights": no_of_week_nights,
                "type_of_meal_plan": type_of_meal_plan,
                "required_car_parking_space": 1 if required_car_parking_space == "Yes" else 0,
                "room_type_reserved": room_type_reserved,
                "lead_time": lead_time,
                "arrival_year": arrival_year,
                "arrival_month": arrival_month,
                "arrival_date": arrival_date,
                "market_segment_type": market_segment_type,
                "repeated_guest": 1 if repeated_guest == "Yes" else 0,
                "no_of_previous_cancellations": no_of_previous_cancellations,
                "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
                "avg_price_per_room": avg_price_per_room,
                "no_of_special_requests": no_of_special_requests,
            }

            input_df = engineer_features(pd.DataFrame([raw_input]))
            proba = pipeline.predict_proba(input_df)[0][1]
            prediction = pipeline.predict(input_df)[0]
            risk_label, risk_emoji = get_risk_tier(proba)

            st.divider()
            st.subheader("Result")

            risk_pct = proba * 100
            result_col1, result_col2 = st.columns([2, 1])
            with result_col1:
                if prediction == 1:
                    st.error(f"⚠️ High risk of cancellation — estimated probability: **{risk_pct:.1f}%**")
                else:
                    st.success(f"✅ Low risk of cancellation — estimated probability: **{risk_pct:.1f}%**")
                st.progress(min(max(proba, 0.0), 1.0))
            with result_col2:
                st.metric("Risk Tier", f"{risk_emoji} {risk_label}")

            # --- SHAP explanation ---
            contrib_df = get_shap_contributions(input_df)
            if contrib_df is not None:
                st.subheader("Why this prediction?")
                st.caption("Top factors driving this specific booking's risk score (SHAP values).")
                chart_df = contrib_df.set_index("feature")[["shap_value"]]
                st.bar_chart(chart_df)
                st.dataframe(contrib_df, hide_index=True, use_container_width=True)

            with st.expander("See the exact input sent to the model"):
                st.dataframe(input_df.T.rename(columns={0: "value"}))

            # --- Session history ---
            st.session_state.history.append({
                "Lead Time": lead_time,
                "Price/Room": avg_price_per_room,
                "Market Segment": market_segment_type,
                "Risk %": round(risk_pct, 1),
                "Tier": f"{risk_emoji} {risk_label}"
            })

    if st.session_state.history:
        st.divider()
        st.subheader("This Session's Predictions")
        st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.caption(
        "This tool provides a statistical estimate based on historical booking patterns. "
        "It should support, not replace, staff judgement on individual reservations."
    )

# =======================================================================
# TAB 2: BATCH PREDICTION
# =======================================================================
with tab_batch:
    st.subheader("Batch Prediction")
    st.write(
        "Upload a CSV with the same raw columns as the training dataset "
        "(no need to include `Booking_ID` or `booking_status`) to get predictions "
        "for many bookings at once."
    )

    required_cols = [c for c in column_info["columns"] if c not in
                      ["total_nights", "total_guests", "total_previous_bookings",
                       "prior_cancel_rate", "has_special_requests", "is_near_holiday"]]
    st.caption("Required columns: " + ", ".join(required_cols))

    uploaded_file = st.file_uploader("Upload booking CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            batch_df = None

        if batch_df is not None:
            missing = [c for c in required_cols if c not in batch_df.columns]
            if missing:
                st.error(f"The uploaded file is missing required columns: {', '.join(missing)}")
            else:
                with st.spinner("Scoring bookings..."):
                    engineered = engineer_features(batch_df)
                    probas = pipeline.predict_proba(engineered[column_info["columns"]])[:, 1]
                    preds = pipeline.predict(engineered[column_info["columns"]])

                    results = batch_df.copy()
                    results["cancellation_probability"] = np.round(probas * 100, 1)
                    results["predicted_status"] = np.where(preds == 1, "Likely to Cancel", "Likely to Keep")
                    results["risk_tier"] = [get_risk_tier(p)[0] for p in probas]

                st.success(f"Scored {len(results)} bookings.")
                st.dataframe(results, use_container_width=True)

                csv_buffer = io.StringIO()
                results.to_csv(csv_buffer, index=False)
                st.download_button(
                    "Download results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="cancellation_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.subheader("Batch Summary")
                tier_counts = results["risk_tier"].value_counts()
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("🔴 High Risk", int(tier_counts.get("High", 0)))
                sc2.metric("🟡 Medium Risk", int(tier_counts.get("Medium", 0)))
                sc3.metric("🟢 Low Risk", int(tier_counts.get("Low", 0)))

# =======================================================================
# TAB 3: INSIGHTS
# =======================================================================
with tab_insights:
    st.subheader("Historical Cancellation Insights")

    if dashboard_data is None:
        st.info(
            "Insights are not available yet. Run the `dashboard_data.pkl` export cell "
            "in the notebook and place the file in this app's folder to enable this tab."
        )
    else:
        m1, m2 = st.columns(2)
        m1.metric("Total Historical Bookings", f"{dashboard_data['total_bookings']:,}")
        m2.metric("Overall Cancellation Rate", f"{dashboard_data['overall_cancellation_rate']*100:.1f}%")

        st.divider()
        ic1, ic2 = st.columns(2)
        with ic1:
            st.write("**Cancellation rate by market segment**")
            seg = pd.Series(dashboard_data["cancellation_by_segment"]).sort_values(ascending=False) * 100
            st.bar_chart(seg)
        with ic2:
            st.write("**Cancellation rate by room type**")
            room = pd.Series(dashboard_data["cancellation_by_room"]).sort_values(ascending=False) * 100
            st.bar_chart(room)

        ic3, ic4 = st.columns(2)
        with ic3:
            st.write("**Cancellation rate by meal plan**")
            meal = pd.Series(dashboard_data["cancellation_by_meal"]).sort_values(ascending=False) * 100
            st.bar_chart(meal)
        with ic4:
            st.write("**Cancellation rate by lead-time bucket**")
            lt = pd.Series(dashboard_data["cancellation_by_leadtime"]) * 100
            st.bar_chart(lt)

# =======================================================================
# TAB 4: MODEL PERFORMANCE
# =======================================================================
with tab_performance:
    st.subheader("Model Performance")

    if dashboard_data is None:
        st.info(
            "Performance details are not available yet. Run the `dashboard_data.pkl` export "
            "cell in the notebook and place the file in this app's folder to enable this tab."
        )
    else:
        st.write("**Model comparison (test set)**")
        comp_df = pd.DataFrame(dashboard_data["model_comparison"])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        pc1, pc2 = st.columns(2)
        with pc1:
            st.write(f"**Confusion Matrix — {best_model_name}**")
            cm = np.array(dashboard_data["confusion_matrix"])
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: Not Cancelled", "Actual: Cancelled"],
                columns=["Predicted: Not Cancelled", "Predicted: Cancelled"]
            )
            st.dataframe(cm_df, use_container_width=True)

        with pc2:
            st.write("**Classification Report**")
            report = dashboard_data["classification_report"]
            report_df = pd.DataFrame(report).T.round(3)
            st.dataframe(report_df, use_container_width=True)

        st.divider()
        st.write("**Top Feature Importances**")
        imp_df = pd.DataFrame(dashboard_data["feature_importance"]).set_index("feature")
        st.bar_chart(imp_df["importance"])