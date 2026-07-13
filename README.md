# 🏨 Hotel Reservation Cancellation Prediction

A machine learning system that predicts whether a hotel booking will be cancelled, identifies the key drivers of cancellation risk, and delivers actionable recommendations — deployed as an interactive Streamlit web application.

**Live demo:** [Add your Streamlit Cloud URL here once deployed]

---

## Project Overview

Hotel cancellations disrupt revenue forecasting, staffing, and inventory planning. This project builds an end-to-end machine learning pipeline that:

- **Predicts** the probability that a given reservation will be cancelled
- **Explains** which factors drive that risk, both globally and for individual bookings
- **Recommends** operational actions the hotel can take to reduce cancellation impact

The project was completed as part of the AIMS Senegal MSc capstone (Group 4).

---

## Dataset

The dataset contains 36,275 hotel bookings with 19 features, including guest details, stay duration, booking lead time, room type, market segment, pricing, and booking history. The target variable, `booking_status`, indicates whether a reservation was cancelled (32.8%) or not (67.2%).

| Category | Examples |
|---|---|
| Guest details | Adults, children, repeated guest status |
| Stay details | Weekend/week nights, room type, meal plan |
| Booking details | Lead time, arrival date, market segment |
| Pricing | Average price per room |
| History | Previous cancellations, previous completed bookings |

---

## Methodology

### 1. Exploratory Data Analysis
Investigated cancellation patterns across lead time, price, market segment, room type, and seasonality to identify likely predictive signals before modeling.

### 2. Feature Engineering
Beyond the raw dataset, the following features were engineered:

| Feature | Description |
|---|---|
| `total_nights` | Weekend nights + week nights |
| `total_guests` | Adults + children |
| `total_previous_bookings` | Previous cancellations + previous completed bookings |
| `prior_cancel_rate` | Share of a guest's past bookings that were cancelled |
| `has_special_requests` | Whether the guest made at least one special request |
| `is_near_holiday` | Whether the arrival date falls within ±3 days of a public holiday (external data enrichment via the `holidays` library, assuming a Portuguese market given euro-denominated pricing) |

### 3. Model Development
Six classifiers were trained and compared on a held-out test set: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, and XGBoost. The top two performers by F1-score were carried forward into hyperparameter tuning via `RandomizedSearchCV` (20 iterations, 5-fold cross-validation).

### 4. Model Selection
**Random Forest** was selected as the final model based on test-set performance.

| Metric | Score |
|---|---|
| Accuracy | 90.5% |
| Precision | 0.89 |
| Recall | 0.82 |
| F1-Score | 0.85 |
| ROC-AUC | 0.96 |

5-fold cross-validation confirmed stability (mean F1 = 0.837, std = 0.007), indicating the model generalizes consistently rather than overfitting to one particular data split.

### 5. Feature Impact Analysis
The strongest predictors of cancellation, by feature importance:

1. **Lead time** (~30%) — bookings made far in advance carry substantially higher risk
2. **Average price per room** (~15%) — higher-priced bookings are more cancellation-prone
3. **Arrival date / month** (~17% combined) — seasonal effects on cancellation likelihood
4. **Special requests** — guests with requests cancel less, signaling stronger commitment
5. **Market segment** — online bookings cancel more than corporate or offline bookings

### 6. Recommendations
- Introduce tiered deposits or confirmation reminders scaled to lead time
- Apply stricter cancellation policies to premium-priced rooms
- Cross-reference high-risk arrival periods against the booking calendar for seasonal planning
- Deprioritize overbooking-risk management for guests with special requests
- Add verification steps (e.g. card confirmation) for online bookings specifically
- Use the model's probability score to triage bookings into risk tiers rather than relying on a binary flag alone

---

## Deployment: Streamlit Application

The trained pipeline is deployed as a multi-tab Streamlit application:

| Tab | Functionality |
|---|---|
| 🔮 **Predict** | Enter booking details to get a cancellation probability, risk tier (Low/Medium/High), and a SHAP-based explanation of the top factors driving that specific prediction |
| 📁 **Batch Prediction** | Upload a CSV of multiple bookings and receive predictions for all of them, downloadable as a results file |
| 📊 **Insights** | Historical cancellation patterns by market segment, room type, meal plan, and lead-time bucket |
| 📈 **Model Performance** | Confusion matrix, classification report, model comparison table, and feature importance chart |

---

## Repository Structure

```
├── app.py                                              # Streamlit deployment application
├── requirements.txt                                    # Python dependencies
├── Hotel_Reservation_Cancellation_Prediction.ipynb     # Full analysis & modeling notebook
├── best_model_pipeline.pkl                             # Trained pipeline (preprocessing + model)
├── column_info.pkl                                     # Feature metadata for the app's input form
├── dashboard_data.pkl                                  # Precomputed summaries for the Insights/Performance tabs
└── README.md
```

---

## Running Locally

```bash
# Clone the repository
git clone https://github.com/wandiyajames-arch/hotel-cancellation-prediction.git
cd hotel-cancellation-prediction

# Create and activate a virtual environment
python3 -m venv hotel
source hotel/bin/activate        # Windows: hotel\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Tech Stack

- **Modeling:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Deployment:** Streamlit
- **Data processing:** pandas, NumPy
- **External enrichment:** `holidays` library

---

## Authors

Group 4 — AIMS Senegal MSc Big Data / Data Science Capstone project
Supervised by IIP program

---

## License

This project was developed for academic purposes as part of the Industry Immersion capstone program.
