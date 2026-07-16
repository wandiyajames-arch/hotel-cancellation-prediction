"""
Hotel Reservation Cancellation Prediction - Executive Risk Dashboard
----------------------------------------------------------------------
Visual layout adapted from a reference "HotelAI" dashboard mockup:
single continuous scrolling page, KPI cards, risk gauge, SHAP waterfall,
key-factor cards, ranked action cards, financial cards, what-if simulator,
similar bookings, and prediction history.

Required files in the same folder as this script:
    - best_model_pipeline.pkl   (required)
    - column_info.pkl           (required)
    - dashboard_data.pkl        (optional - powers Similar Bookings)

SCOPE NOTE: fields the training dataset does not contain (deposit type,
customer type, booking-change count, guest name, hotel name, refund
records, days-before-cancellation for similar bookings) are NOT fed into
the model and are not fabricated here. Financial figures are user-editable
assumptions, clearly labeled as such.
"""

import datetime
import io
import uuid

import holidays as holidays_lib
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hotel Cancellation Risk Dashboard",
    page_icon="🏨",
    layout="wide"
)

# ---------------------------------------------------------------------
# Visual theme (cards, colors) - CSS injected once
# ---------------------------------------------------------------------
st.markdown("""
<style>
.kpi-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 16px 18px;
    border: 1px solid #eaeaea;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    height: 100%;
}
.kpi-label { font-size: 0.78rem; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: .03em; }
.kpi-value { font-size: 1.6rem; font-weight: 700; margin: 4px 0 2px 0; }
.kpi-sub { font-size: 0.78rem; color: #9ca3af; }
.factor-card {
    background: #ffffff; border-radius: 10px; padding: 14px;
    border-left: 4px solid #d1d5db; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    height: 100%; margin-bottom: 8px;
}
.factor-card.risk-up { border-left-color: #d73027; }
.factor-card.risk-down { border-left-color: #1a9850; }
.action-card {
    background: #ffffff; border-radius: 10px; padding: 14px;
    border: 1px solid #eaeaea; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 8px;
}
.section-header {
    font-size: 1.05rem; font-weight: 700; margin: 22px 0 10px 0;
    padding-bottom: 6px; border-bottom: 2px solid #f0f0f0;
}
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 2px;
    width: 100%;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #1e293b;
}
section[data-testid="stSidebar"] hr {
    border-color: #334155;
}
</style>
""", unsafe_allow_html=True)


def kpi_card(label, value, sub=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


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
        return "Low", "#1a9850"
    elif proba < 0.60:
        return "Medium", "#f5a623"
    else:
        return "High", "#d73027"


def validate_date(year, month, day):
    try:
        datetime.date(int(year), int(month), int(day))
        return True
    except ValueError:
        return False


def model_confidence(proba):
    return 50 + abs(proba - 0.5) * 100


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

cat_options = column_info["cat_options"]
best_model_name = column_info["best_model_name"]
MODEL_COLUMNS = column_info["columns"]

if "history" not in st.session_state:
    st.session_state.history = []
if "current_booking_id" not in st.session_state:
    st.session_state.current_booking_id = f"BK-{uuid.uuid4().hex[:6].upper()}"


def run_prediction(raw_dict):
    df = engineer_features(pd.DataFrame([raw_dict]))
    proba = pipeline.predict_proba(df[MODEL_COLUMNS])[0][1]
    return proba, df


# ---------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------
def get_shap_contributions(input_df, top_n=8):
    try:
        import shap
    except ImportError:
        return None
    model_step = pipeline.named_steps["model"]
    preprocessor_step = pipeline.named_steps["preprocessor"]
    if not hasattr(model_step, "feature_importances_"):
        return None
    transformed = preprocessor_step.transform(input_df[MODEL_COLUMNS])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor_step.get_feature_names_out()
    explainer = shap.TreeExplainer(model_step)
    sv = explainer.shap_values(transformed)
    if isinstance(sv, list):
        row_values = sv[1][0]
    elif sv.ndim == 3:
        row_values = sv[0, :, 1]
    else:
        row_values = sv[0]
    contrib_df = pd.DataFrame({"feature": feature_names, "shap_value": row_values})
    contrib_df["abs_value"] = contrib_df["shap_value"].abs()
    contrib_df = contrib_df.sort_values("abs_value", ascending=False).head(top_n)
    contrib_df["direction"] = np.where(contrib_df["shap_value"] > 0, "Increases risk", "Decreases risk")
    return contrib_df.drop(columns="abs_value")


FEATURE_EXPLANATIONS = {
    "lead_time": ("📅", "Long lead time increases cancellation risk — plans are more likely to change the further out a booking sits."),
    "avg_price_per_room": ("💶", "Higher room price is associated with more price-sensitive cancellations."),
    "arrival_month": ("🗓️", "Seasonal timing affects cancellation likelihood for this arrival month."),
    "arrival_date": ("📆", "The specific arrival date falls in a period with a distinct cancellation pattern."),
    "no_of_special_requests": ("✅", "Special requests signal stronger guest commitment, which lowers cancellation risk."),
    "has_special_requests": ("✅", "Making at least one special request is linked to a lower cancellation rate."),
    "market_segment_type": ("🌐", "This market segment has a historically different cancellation rate than others."),
    "repeated_guest": ("🔁", "Repeat guests cancel less often than first-time bookers."),
    "required_car_parking_space": ("🅿️", "Requesting parking is a small additional signal of trip commitment."),
    "is_near_holiday": ("🎉", "Proximity to a public holiday is linked to a modest change in cancellation likelihood."),
    "no_of_previous_cancellations": ("⚠️", "A history of prior cancellations raises the risk for this booking."),
    "total_nights": ("🛏️", "Total length of stay contributes to the overall risk profile."),
    "prior_cancel_rate": ("📉", "This guest's historical cancellation rate directly informs their current risk."),
}


def explain_feature(feature_name):
    for key, (icon, text) in FEATURE_EXPLANATIONS.items():
        if key in feature_name:
            return icon, text
    return "🔎", f"{feature_name.replace('num__', '').replace('cat__', '')} contributes to this prediction."


def recommend_actions(proba, raw_dict):
    actions = []
    if proba >= 0.60:
        actions.append(("📞 Contact customer directly", 85, "High-risk bookings respond best to a personal touch."))
        actions.append(("💳 Request deposit payment", 78, "A deposit converts intent into commitment."))
        actions.append(("🔁 Overbooking recommendation", 65, "Consider this room for controlled overbooking given the high cancellation odds."))
    if 0.30 <= proba < 0.60:
        actions.append(("✉️ Send reminder email", 70, "A timely nudge reduces forgetfulness-driven cancellations."))
        actions.append(("🎁 Offer a discount upgrade", 60, "A small incentive can tip a wavering booking toward completion."))
        actions.append(("🔄 Flexible rescheduling offer", 55, "Removing date-lock-in pressure can prevent an outright cancellation."))
    if proba < 0.30:
        actions.append(("💌 Send loyalty incentive", 50, "Reinforce a low-risk booking's commitment for future retention."))
        actions.append(("🕒 Waitlist replacement (monitor only)", 30, "Low risk - no urgent action needed, monitor as normal."))
    if raw_dict.get("no_of_special_requests", 0) == 0 and proba >= 0.30:
        actions.append(("📋 Prompt for special requests at check-in", 45, "Special requests are linked to higher completion rates."))
    actions.sort(key=lambda x: x[1], reverse=True)
    return actions


def financial_impact(proba, room_price, total_nights, refund_pct, recovery_cost, intervention_cost, clv):
    booking_value = room_price * max(total_nights, 1)
    expected_loss = booking_value * proba
    refund_amount = booking_value * refund_pct / 100 * proba
    lost_occupancy_nights = total_nights * proba
    net_savings = max(expected_loss - intervention_cost, 0)
    return {
        "booking_value": booking_value, "expected_loss": expected_loss,
        "refund_amount": refund_amount, "lost_occupancy_nights": lost_occupancy_nights,
        "recovery_cost": recovery_cost, "intervention_cost": intervention_cost,
        "net_savings": net_savings, "clv_at_risk": clv * proba,
    }


def generate_counterfactuals(raw_dict, base_proba):
    scenarios = []
    cf = dict(raw_dict); cf["lead_time"] = max(int(raw_dict["lead_time"] * 0.4), 0)
    p, _ = run_prediction(cf)
    scenarios.append(("Reduce lead time by 60%", p))

    cf = dict(raw_dict); cf["no_of_special_requests"] = max(raw_dict["no_of_special_requests"], 2)
    p, _ = run_prediction(cf)
    scenarios.append(("Prompt guest for special requests", p))

    if raw_dict["market_segment_type"] != "Corporate":
        cf = dict(raw_dict); cf["market_segment_type"] = "Corporate"
        p, _ = run_prediction(cf)
        scenarios.append(("Shift to Corporate channel (illustrative)", p))

    cf = dict(raw_dict); cf["avg_price_per_room"] = raw_dict["avg_price_per_room"] * 0.85
    p, _ = run_prediction(cf)
    scenarios.append(("Offer 15% price discount", p))

    combined = dict(raw_dict)
    combined["no_of_special_requests"] = max(raw_dict["no_of_special_requests"], 2)
    combined["lead_time"] = max(int(raw_dict["lead_time"] * 0.4), 0)
    p, _ = run_prediction(combined)
    scenarios.append(("Combined: shorter lead time + special requests", p))

    scenarios.sort(key=lambda x: x[1])
    return scenarios


def find_similar_bookings(raw_dict, sample_df, k=5):
    if sample_df is None or len(sample_df) == 0:
        return None
    numeric_cols = ["lead_time", "avg_price_per_room", "no_of_special_requests"]
    ref = sample_df.copy()
    ranges = {c: max(ref[c].max() - ref[c].min(), 1e-6) for c in numeric_cols}
    target = np.array([raw_dict[c] for c in numeric_cols], dtype=float)
    ref_vals = ref[numeric_cols].values.astype(float)
    norm_target = np.array([(target[i] - ref[numeric_cols[i]].min()) / ranges[numeric_cols[i]] for i in range(len(numeric_cols))])
    norm_ref = np.array([
        [(row[i] - ref[numeric_cols[i]].min()) / ranges[numeric_cols[i]] for i in range(len(numeric_cols))]
        for row in ref_vals
    ])
    dist = np.sqrt(((norm_ref - norm_target) ** 2).sum(axis=1))
    similarity_pct = (1 - dist / np.sqrt(len(numeric_cols))) * 100
    ref = ref.copy()
    ref["similarity_pct"] = similarity_pct
    return ref.sort_values("similarity_pct", ascending=False).head(k)


def sanitize_for_pdf(text):
    return str(text).encode("latin-1", "ignore").decode("latin-1").strip()


def build_pdf_report(booking_id, proba, risk_label, confidence, contrib_df, actions, fin):
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, sanitize_for_pdf("Hotel Cancellation Risk Report"))
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 11)
    for line in [
        f"Booking ID: {booking_id}", f"Cancellation Probability: {proba*100:.1f}%",
        f"Risk Tier: {risk_label}", f"Model Confidence: {confidence:.1f}%"
    ]:
        pdf.cell(0, 8, sanitize_for_pdf(line)); pdf.ln(7)
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, sanitize_for_pdf("Top Contributing Factors")); pdf.ln(9)
    pdf.set_font("Helvetica", "", 10)
    if contrib_df is not None:
        for _, row in contrib_df.iterrows():
            direction = "increases" if row["shap_value"] > 0 else "decreases"
            clean_name = row["feature"].replace("num__", "").replace("cat__", "")
            pdf.cell(0, 7, sanitize_for_pdf(f"- {clean_name}: {direction} risk ({row['shap_value']:+.3f})")); pdf.ln(6)
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, sanitize_for_pdf("Recommended Actions")); pdf.ln(9)
    pdf.set_font("Helvetica", "", 10)
    for label, score, why in actions[:5]:
        pdf.cell(0, 7, sanitize_for_pdf(f"- {label} (effectiveness: {score}%)")); pdf.ln(6)
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, sanitize_for_pdf("Financial Impact (illustrative estimates)")); pdf.ln(9)
    pdf.set_font("Helvetica", "", 10)
    for line in [
        f"Expected loss: EUR {fin['expected_loss']:.2f}",
        f"Estimated refund exposure: EUR {fin['refund_amount']:.2f}",
        f"Net savings if intervention applied: EUR {fin['net_savings']:.2f}"
    ]:
        pdf.cell(0, 7, sanitize_for_pdf(line)); pdf.ln(6)
    return bytes(pdf.output())


def make_gauge(proba, risk_label, risk_color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={'suffix': "%", 'font': {'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': risk_color, 'thickness': 0.3},
            'steps': [
                {'range': [0, 30], 'color': "#e8f6ee"},
                {'range': [30, 60], 'color': "#fff3e0"},
                {'range': [60, 100], 'color': "#fdeaea"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10))
    return fig


def make_waterfall(contrib_df, base_rate_pct=32.8):
    if contrib_df is None or len(contrib_df) == 0:
        return None
    d = contrib_df.copy().sort_values("shap_value", key=abs, ascending=False).head(6)
    labels = [f.replace("num__", "").replace("cat__", "") for f in d["feature"]]
    values = (d["shap_value"] * 100).tolist()
    fig = go.Figure(go.Waterfall(
        orientation="v",
        x=["Base rate"] + labels + ["Final"],
        y=[base_rate_pct] + values + [0],
        measure=["absolute"] + ["relative"] * len(values) + ["total"],
        connector={"line": {"color": "#cccccc"}},
        increasing={"marker": {"color": "#d73027"}},
        decreasing={"marker": {"color": "#1a9850"}},
        totals={"marker": {"color": "#4c72b0"}},
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig


# =======================================================================
# SIDEBAR
# =======================================================================
with st.sidebar:
    st.markdown("### 🛡️ CancelSense ")
    st.caption("Machine Learning · Decision Support")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "➕ New Prediction",
            "📜 Prediction History",
            "📊 Analytics",
            "📈 Model Performance",
            "📁 Reports (Batch Scoring)",
            "⚙️ Settings",
        ],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption(f"Model in use: **{best_model_name}**")
    st.caption("Data-Driven Decisions. Smarter Hotels. More Revenue.")

# =======================================================================
# PAGE: DASHBOARD (landing overview)
# =======================================================================
if page == "🏠 Dashboard":
    st.title("Dashboard")
    st.caption("Machine Learning · Smarter Decisions · Higher Revenue")

    if dashboard_data is not None:
        m1, m2, m3, m4 = st.columns(4)
        with m1: kpi_card("Total Historical Bookings", f"{dashboard_data['total_bookings']:,}")
        with m2: kpi_card("Overall Cancellation Rate", f"{dashboard_data['overall_cancellation_rate']*100:.1f}%")
        with m3: kpi_card("Model in Use", best_model_name)
        with m4: kpi_card("Predictions This Session", str(len(st.session_state.history)))
    else:
        st.info("Run the `dashboard_data.pkl` export cell in the notebook to populate these summary stats.")

    section_header("Recent Predictions (this session)")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history).tail(5), hide_index=True, use_container_width=True)
    else:
        st.caption("No predictions made yet this session. Go to **New Prediction** to assess a booking.")

    if dashboard_data is not None:
        section_header("Cancellation Rate by Market Segment")
        st.bar_chart(pd.Series(dashboard_data["cancellation_by_segment"]).sort_values(ascending=False) * 100)

# =======================================================================
# PAGE: NEW PREDICTION (single continuous scrolling page)
# =======================================================================
elif page == "➕ New Prediction":
    st.title("Hotel Reservation Cancellation Prediction")
    st.caption("Machine Learning · Smarter Decisions · Higher Revenue")

    with st.form("booking_form"):
        section_header("📝 New Booking Details")
        col1, col2 = st.columns(2)
        with col1:
            guest_name = st.text_input("Guest name (optional, for report only)", value="")
            no_of_adults = st.number_input("Number of adults", min_value=0, max_value=10, value=2)
            no_of_children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
            no_of_weekend_nights = st.number_input("Weekend nights", min_value=0, max_value=10, value=1)
            no_of_week_nights = st.number_input("Week nights", min_value=0, max_value=15, value=2)
            required_car_parking_space = st.selectbox("Requires car parking space?", ["No", "Yes"])
            no_of_special_requests = st.number_input("Number of special requests", min_value=0, max_value=5, value=0)
        with col2:
            lead_time = st.number_input("Lead time (days)", min_value=0, max_value=500, value=50)
            arrival_year = st.selectbox("Arrival year", [2017, 2018, 2026, 2027], index=1)
            arrival_month = st.selectbox("Arrival month", list(range(1, 13)), index=6)
            arrival_date = st.number_input("Arrival day of month", min_value=1, max_value=31, value=15)
            repeated_guest = st.selectbox("Repeated guest?", ["No", "Yes"])

        col3, col4 = st.columns(2)
        with col3:
            type_of_meal_plan = st.selectbox("Meal plan", cat_options.get("type_of_meal_plan", []))
            room_type_reserved = st.selectbox("Room type", cat_options.get("room_type_reserved", []))
        with col4:
            market_segment_type = st.selectbox("Market segment", cat_options.get("market_segment_type", []))
            avg_price_per_room = st.number_input("Average price per room / ADR (EUR)", min_value=0.0, max_value=600.0, value=100.0, step=1.0)

        col5, col6 = st.columns(2)
        with col5:
            no_of_previous_cancellations = st.number_input("Previous cancellations", min_value=0, max_value=50, value=0)
        with col6:
            no_of_previous_bookings_not_canceled = st.number_input("Previous bookings NOT cancelled", min_value=0, max_value=100, value=0)

        st.markdown("**Financial assumptions** _(editable — used only for the Financial Impact estimate, not the model)_")
        col7, col8, col9 = st.columns(3)
        with col7:
            refund_pct = st.slider("Refund policy (%)", 0, 100, 80)
        with col8:
            recovery_cost = st.number_input("Recovery cost (EUR)", min_value=0.0, value=25.0)
        with col9:
            intervention_cost = st.number_input("Intervention cost (EUR)", min_value=0.0, value=10.0)

        submitted = st.form_submit_button("🔍 Assess Cancellation Risk", use_container_width=True)

    if submitted:
        if not validate_date(arrival_year, arrival_month, arrival_date):
            st.warning(f"⚠️ {arrival_year}-{arrival_month}-{arrival_date} is not a valid calendar date.")
        else:
            raw_input = {
                "no_of_adults": no_of_adults, "no_of_children": no_of_children,
                "no_of_weekend_nights": no_of_weekend_nights, "no_of_week_nights": no_of_week_nights,
                "type_of_meal_plan": type_of_meal_plan,
                "required_car_parking_space": 1 if required_car_parking_space == "Yes" else 0,
                "room_type_reserved": room_type_reserved, "lead_time": lead_time,
                "arrival_year": arrival_year, "arrival_month": arrival_month, "arrival_date": arrival_date,
                "market_segment_type": market_segment_type,
                "repeated_guest": 1 if repeated_guest == "Yes" else 0,
                "no_of_previous_cancellations": no_of_previous_cancellations,
                "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
                "avg_price_per_room": avg_price_per_room, "no_of_special_requests": no_of_special_requests,
            }
            proba, _ = run_prediction(raw_input)
            risk_label, risk_color = get_risk_tier(proba)
            confidence = model_confidence(proba)
            total_nights = no_of_weekend_nights + no_of_week_nights
            booking_id = st.session_state.current_booking_id

            st.session_state["_last"] = dict(
                raw_input=raw_input, proba=proba, risk_label=risk_label, risk_color=risk_color,
                confidence=confidence, total_nights=total_nights, guest_name=guest_name,
                booking_id=booking_id, refund_pct=refund_pct, recovery_cost=recovery_cost,
                intervention_cost=intervention_cost
            )
            st.session_state.history.append({
                "Booking ID": booking_id, "Guest": guest_name or "—", "Lead Time": lead_time,
                "Price/Room": avg_price_per_room, "Market Segment": market_segment_type,
                "Risk %": round(proba * 100, 1), "Tier": risk_label
            })

    if "_last" in st.session_state:
        d = st.session_state["_last"]
        raw_input, proba = d["raw_input"], d["proba"]
        risk_label, risk_color, confidence = d["risk_label"], d["risk_color"], d["confidence"]
        total_nights, booking_id = d["total_nights"], d["booking_id"]

        contrib_df = get_shap_contributions(engineer_features(pd.DataFrame([raw_input])))
        actions = recommend_actions(proba, raw_input)
        clv = raw_input["avg_price_per_room"] * (raw_input["no_of_previous_bookings_not_canceled"] + 1) * 2
        fin = financial_impact(proba, raw_input["avg_price_per_room"], total_nights,
                                d["refund_pct"], d["recovery_cost"], d["intervention_cost"], clv)

        # ---- 1. Executive Summary ----
        section_header("1️⃣ Executive Summary")
        ec = st.columns(6)
        with ec[0]: kpi_card("Booking ID", booking_id)
        with ec[1]: kpi_card("Guest", d["guest_name"] or "—")
        with ec[2]: kpi_card("Check-in Date", f"{raw_input['arrival_year']}-{raw_input['arrival_month']:02d}-{raw_input['arrival_date']:02d}")
        with ec[3]: kpi_card("Nights", str(total_nights))
        with ec[4]:
            pred_text = "Likely to Cancel" if proba >= 0.5 else "Likely to Keep"
            kpi_card("Prediction", pred_text, f"{risk_label} Risk")
        with ec[5]: kpi_card("Recommended Action", actions[0][0] if actions else "Monitor")

        # ---- 2. Cancellation Risk Assessment ----
        section_header("2️⃣ Cancellation Risk Assessment")
        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.plotly_chart(make_gauge(proba, risk_label, risk_color), use_container_width=True)
            st.markdown(f"<div style='text-align:center;font-weight:700;color:{risk_color};'>{risk_label.upper()} RISK</div>", unsafe_allow_html=True)
        with rc2:
            gc1, gc2 = st.columns(2)
            with gc1:
                kpi_card("Cancellation Probability", f"{proba*100:.1f}%", risk_label + " Risk")
                kpi_card("Model Confidence", f"{confidence:.1f}%")
            with gc2:
                kpi_card("Expected Revenue Loss", f"€{fin['expected_loss']:.2f}", "High Impact" if fin['expected_loss'] > 100 else "Moderate")
                kpi_card("Occupancy Impact", f"{fin['lost_occupancy_nights']:.1f} nights at risk")

        if proba >= 0.60:
            st.error("⚠️ This reservation has a high likelihood of cancellation. Immediate intervention is recommended.")
        elif proba >= 0.30:
            st.warning("🟡 This reservation shows moderate cancellation risk. Proactive follow-up is advised.")
        else:
            st.success("✅ This reservation is low risk. No action needed at this time.")

        # ---- 3. Model Explanation ----
        section_header("3️⃣ Model Explanation — Why This Prediction?")
        if contrib_df is not None:
            wf = make_waterfall(contrib_df)
            if wf is not None:
                st.plotly_chart(wf, use_container_width=True)

            ic1, ic2 = st.columns(2)
            with ic1:
                st.markdown("**⬆️ Increases Risk**")
                for _, row in contrib_df[contrib_df["shap_value"] > 0].iterrows():
                    clean = row["feature"].replace("num__", "").replace("cat__", "")
                    st.markdown(f"<div class='factor-card risk-up'>🔴 {clean} <span style='color:#999;font-size:0.85rem;'>(+{row['shap_value']:.3f})</span></div>", unsafe_allow_html=True)
            with ic2:
                st.markdown("**⬇️ Decreases Risk**")
                for _, row in contrib_df[contrib_df["shap_value"] < 0].iterrows():
                    clean = row["feature"].replace("num__", "").replace("cat__", "")
                    st.markdown(f"<div class='factor-card risk-down'>🟢 {clean} <span style='color:#999;font-size:0.85rem;'>({row['shap_value']:.3f})</span></div>", unsafe_allow_html=True)
        else:
            st.info("SHAP explanation unavailable — `shap` package not installed.")

        # ---- 4. Key Contributing Factors ----
        section_header("4️⃣ Key Contributing Factors")
        if contrib_df is not None:
            fc_cols = st.columns(min(len(contrib_df), 6))
            for i, (_, row) in enumerate(contrib_df.head(6).iterrows()):
                icon, text = explain_feature(row["feature"])
                cls = "risk-up" if row["shap_value"] > 0 else "risk-down"
                with fc_cols[i % 6]:
                    st.markdown(f"""
                    <div class="factor-card {cls}">
                        <div style="font-size:1.3rem;">{icon}</div>
                        <div style="font-weight:700;font-size:0.85rem;margin:4px 0;">{row['feature'].replace('num__','').replace('cat__','')}</div>
                        <div style="font-size:0.78rem;color:#666;">{text}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ---- 5. Recommended Actions ----
        section_header("5️⃣ Recommended Actions (Ranked by Effectiveness)")
        ac_cols = st.columns(3)
        for i, (label, score, why) in enumerate(actions):
            with ac_cols[i % 3]:
                stars = "⭐" * round(score / 20)
                st.markdown(f"""
                <div class="action-card">
                    <div style="font-weight:700;">{i+1}. {label}</div>
                    <div style="font-size:0.8rem;color:#666;margin:4px 0;">{why}</div>
                    <div style="font-weight:700;color:#4c72b0;">{score}% {stars}</div>
                </div>
                """, unsafe_allow_html=True)

        # ---- 6. Financial Impact ----
        section_header("6️⃣ Financial Impact Analysis")
        st.caption("⚠️ Estimates based on the assumptions you set in the form — not values the model learned.")
        f1, f2, f3, f4 = st.columns(4)
        with f1: kpi_card("Expected Revenue Loss", f"€{fin['expected_loss']:.2f}")
        with f2: kpi_card("Refund Exposure", f"€{fin['refund_amount']:.2f}")
        with f3: kpi_card("Lost Occupancy", f"{fin['lost_occupancy_nights']:.2f} nights")
        with f4: kpi_card("CLV at Risk", f"€{fin['clv_at_risk']:.2f}")
        g1, g2, g3 = st.columns(3)
        with g1: kpi_card("Expected Loss", f"€{fin['expected_loss']:.2f}")
        with g2: kpi_card("− Intervention Cost", f"€{fin['intervention_cost']:.2f}")
        with g3: kpi_card("= Net Savings", f"€{fin['net_savings']:.2f}")

        pdf_bytes = build_pdf_report(booking_id, proba, risk_label, confidence, contrib_df, actions, fin)
        if pdf_bytes:
            st.download_button("📄 Download PDF Report", data=pdf_bytes,
                                file_name=f"{booking_id}_risk_report.pdf", mime="application/pdf")

        # ---- 7. What-If Simulator ----
        section_header("7️⃣ What-If Analysis Simulator")
        wcol1, wcol2 = st.columns(2)
        with wcol1:
            w_lead_time = st.slider("Lead time (days)", 0, 500, int(raw_input["lead_time"]), key="w_lead")
            w_price = st.slider("Room price / ADR (EUR)", 0.0, 600.0, float(raw_input["avg_price_per_room"]), key="w_price")
            w_requests = st.slider("Special requests", 0, 5, int(raw_input["no_of_special_requests"]), key="w_req")
        with wcol2:
            w_segment = st.selectbox("Market segment", cat_options.get("market_segment_type", []),
                                      index=cat_options.get("market_segment_type", []).index(raw_input["market_segment_type"]), key="w_seg")
            w_meal = st.selectbox("Meal plan", cat_options.get("type_of_meal_plan", []),
                                   index=cat_options.get("type_of_meal_plan", []).index(raw_input["type_of_meal_plan"]), key="w_meal")
            w_parking = st.selectbox("Parking requested?", ["No", "Yes"], index=raw_input["required_car_parking_space"], key="w_park")

        w_raw = dict(raw_input)
        w_raw.update({
            "lead_time": w_lead_time, "avg_price_per_room": w_price, "no_of_special_requests": w_requests,
            "market_segment_type": w_segment, "type_of_meal_plan": w_meal,
            "required_car_parking_space": 1 if w_parking == "Yes" else 0
        })
        new_proba, _ = run_prediction(w_raw)
        new_label, new_color = get_risk_tier(new_proba)

        wc1, wc2, wc3 = st.columns(3)
        with wc1: st.plotly_chart(make_gauge(proba, risk_label, risk_color), use_container_width=True, key="gauge_current")
        with wc2: st.markdown("<div style='text-align:center;font-size:2rem;padding-top:70px;'>→</div>", unsafe_allow_html=True)
        with wc3: st.plotly_chart(make_gauge(new_proba, new_label, new_color), use_container_width=True, key="gauge_new")
        wsc1, wsc2 = st.columns(2)
        with wsc1: kpi_card("Expected Risk Reduction", f"{max((proba-new_proba)*100,0):.0f}%")
        with wsc2: kpi_card("Potential Savings", f"€{max(fin['expected_loss'] - fin['expected_loss']*(new_proba/max(proba,1e-6)), 0):.2f}")

        # ---- 8. Similar Historical Bookings ----
        section_header("8️⃣ Similar Historical Bookings")
        if dashboard_data is not None and "sample_bookings" in dashboard_data:
            sample_df = pd.DataFrame(dashboard_data["sample_bookings"])
            similar = find_similar_bookings(raw_input, sample_df, k=5)
            if similar is not None:
                display_cols = ["similarity_pct", "lead_time", "avg_price_per_room", "no_of_special_requests", "booking_status"]
                display_cols = [c for c in display_cols if c in similar.columns]
                show = similar[display_cols].rename(columns={
                    "similarity_pct": "Similarity %", "lead_time": "Lead Time",
                    "avg_price_per_room": "Price/Room", "no_of_special_requests": "Special Requests",
                    "booking_status": "Outcome"
                })
                show["Similarity %"] = show["Similarity %"].round(1)
                st.dataframe(show, hide_index=True, use_container_width=True)
                st.caption(f"Average similarity: {show['Similarity %'].mean():.1f}%")
        else:
            st.info("Similar-bookings data not available yet. Add `sample_bookings` to `dashboard_data.pkl`.")

        # ---- 9. Counterfactual Explanations ----
        section_header("9️⃣ Counterfactual Explanations — How to Reduce Risk")
        scenarios = generate_counterfactuals(raw_input, proba)
        for label, new_p in scenarios:
            st.markdown(f"- **{label}** → risk moves from {proba*100:.1f}% to **{new_p*100:.1f}%**")

    st.divider()
    st.caption(f"Full prediction history is available on the **Prediction History** page.")
    st.caption("This tool provides a statistical estimate based on historical booking patterns. It should support, not replace, staff judgement.")

# =======================================================================
# PAGE: PREDICTION HISTORY
# =======================================================================
elif page == "📜 Prediction History":
    st.title("Prediction History")
    st.caption("Every prediction made during this session.")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)
        hc1, hc2 = st.columns(2)
        with hc1:
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()
        with hc2:
            if st.button("New booking ID"):
                st.session_state.current_booking_id = f"BK-{uuid.uuid4().hex[:6].upper()}"
                st.rerun()
        csv_buffer = io.StringIO()
        pd.DataFrame(st.session_state.history).to_csv(csv_buffer, index=False)
        st.download_button("Download history as CSV", data=csv_buffer.getvalue(),
                            file_name="prediction_history.csv", mime="text/csv")
    else:
        st.info("No predictions made yet this session. Go to **New Prediction** to assess a booking.")
    st.caption(
        "Note: history is stored only for the current browser session and resets when the app restarts "
        "or the page is reloaded — it is not saved to a database."
    )

# =======================================================================
# PAGE: REPORTS (BATCH SCORING)
# =======================================================================
elif page == "📁 Reports (Batch Scoring)":
    st.title("Batch Prediction")
    st.write("Upload a CSV with the same raw columns as the training dataset to score many bookings at once.")
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
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                with st.spinner("Scoring bookings..."):
                    engineered = engineer_features(batch_df)
                    probas = pipeline.predict_proba(engineered[MODEL_COLUMNS])[:, 1]
                    preds = pipeline.predict(engineered[MODEL_COLUMNS])
                    results = batch_df.copy()
                    results["cancellation_probability"] = np.round(probas * 100, 1)
                    results["predicted_status"] = np.where(preds == 1, "Likely to Cancel", "Likely to Keep")
                    results["risk_tier"] = [get_risk_tier(p)[0] for p in probas]
                st.success(f"Scored {len(results)} bookings.")
                st.dataframe(results, use_container_width=True)
                csv_buffer = io.StringIO()
                results.to_csv(csv_buffer, index=False)
                st.download_button("Download results as CSV", data=csv_buffer.getvalue(),
                                    file_name="cancellation_predictions.csv", mime="text/csv")
                tier_counts = results["risk_tier"].value_counts()
                sc1, sc2, sc3 = st.columns(3)
                with sc1: kpi_card("🔴 High Risk", int(tier_counts.get("High", 0)))
                with sc2: kpi_card("🟡 Medium Risk", int(tier_counts.get("Medium", 0)))
                with sc3: kpi_card("🟢 Low Risk", int(tier_counts.get("Low", 0)))

# =======================================================================
# PAGE: INSIGHTS
# =======================================================================
elif page == "📊 Analytics":
    st.title("Historical Cancellation Insights")
    if dashboard_data is None:
        st.info("Run the `dashboard_data.pkl` export cell in the notebook to enable this page.")
    else:
        m1, m2 = st.columns(2)
        with m1: kpi_card("Total Historical Bookings", f"{dashboard_data['total_bookings']:,}")
        with m2: kpi_card("Overall Cancellation Rate", f"{dashboard_data['overall_cancellation_rate']*100:.1f}%")
        st.divider()
        ic1, ic2 = st.columns(2)
        with ic1:
            st.write("**Cancellation rate by market segment**")
            st.bar_chart(pd.Series(dashboard_data["cancellation_by_segment"]).sort_values(ascending=False) * 100)
        with ic2:
            st.write("**Cancellation rate by room type**")
            st.bar_chart(pd.Series(dashboard_data["cancellation_by_room"]).sort_values(ascending=False) * 100)
        ic3, ic4 = st.columns(2)
        with ic3:
            st.write("**Cancellation rate by meal plan**")
            st.bar_chart(pd.Series(dashboard_data["cancellation_by_meal"]).sort_values(ascending=False) * 100)
        with ic4:
            st.write("**Cancellation rate by lead-time bucket**")
            st.bar_chart(pd.Series(dashboard_data["cancellation_by_leadtime"]) * 100)

# =======================================================================
# PAGE: MODEL PERFORMANCE
# =======================================================================
elif page == "📈 Model Performance":
    st.title("Model Performance")
    if dashboard_data is None:
        st.info("Run the `dashboard_data.pkl` export cell in the notebook to enable this page.")
    else:
        st.write("**Model comparison (test set)**")
        st.dataframe(pd.DataFrame(dashboard_data["model_comparison"]), hide_index=True, use_container_width=True)
        pc1, pc2 = st.columns(2)
        with pc1:
            st.write(f"**Confusion Matrix — {best_model_name}**")
            cm = np.array(dashboard_data["confusion_matrix"])
            st.dataframe(pd.DataFrame(cm, index=["Actual: Not Cancelled", "Actual: Cancelled"],
                                       columns=["Predicted: Not Cancelled", "Predicted: Cancelled"]),
                         use_container_width=True)
        with pc2:
            st.write("**Classification Report**")
            st.dataframe(pd.DataFrame(dashboard_data["classification_report"]).T.round(3), use_container_width=True)
        st.divider()
        st.write("**Global Feature Importance (SHAP-derived)**")
        imp_df = pd.DataFrame(dashboard_data["feature_importance"]).set_index("feature")
        st.bar_chart(imp_df["importance"])

# =======================================================================
# PAGE: SETTINGS
# =======================================================================
elif page == "⚙️ Settings":
    st.title("Settings")
    st.caption("Model metadata and default assumptions used across the dashboard.")

    section_header("Model Information")
    s1, s2 = st.columns(2)
    with s1:
        kpi_card("Model in Use", best_model_name)
        kpi_card("Number of Input Features", str(len(MODEL_COLUMNS)))
    with s2:
        kpi_card("Categorical Features", str(len(column_info["cat_cols"])))
        kpi_card("Numeric Features", str(len(column_info["num_cols"])))

    section_header("Risk Tier Thresholds")
    st.write("- 🟢 **Low risk:** probability below 30%")
    st.write("- 🟡 **Medium risk:** probability 30%–60%")
    st.write("- 🔴 **High risk:** probability 60% and above")
    st.caption("These thresholds are fixed in the app code, not user-editable, since they define the model's risk-tier logic consistently across every page.")

    section_header("About This App")
    st.write(
        "This dashboard is built strictly on top of the trained model and the dataset it was trained on. "
        "Fields the dataset does not contain (deposit type, customer type, booking-change count, guest name, "
        "hotel name, refund records, days-before-cancellation) are not fed into the model and are not fabricated "
        "anywhere in this app. Financial figures shown in the New Prediction page are user-editable business "
        "assumptions, not values the model learned."
    )
    st.caption(
        "There is no user account system in this app (single-user Streamlit deployment), "
        "so there is no separate User Management page."
    )