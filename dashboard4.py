import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(
    page_title="AI Revenue Recovery",
    page_icon="💳",
    layout="wide"
)

df = pd.read_csv("data/transactions.csv")

X = df.drop(columns=["recovered", "transaction_id"])
y = df["recovered"]

categorical_features = [
    "failure_reason",
    "payment_method",
    "customer_type"
]

numeric_features = [
    "amount",
    "attempts",
    "previous_success_rate",
    "previous_failed_payments",
    "customer_tenure_months",
    "hours_since_failure"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

st.title("AI Revenue Recovery Engine")
st.write("Intelligent payment failure prediction and revenue recovery.")

st.divider()

st.header("Transaction Recovery Predictor")

c1, c2, c3 = st.columns(3)

with c1:

    amount = st.number_input(
        "Transaction Amount",
        min_value=1.0,
        value=7500.0,
        step=100.0,
        key="input_amount"
    )

    failure_reason = st.selectbox(
        "Failure Reason",
        [
            "network_error",
            "insufficient_funds",
            "bank_decline",
            "card_expired",
            "technical_error"
        ],
        key="input_failure_reason"
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "credit_card",
            "debit_card",
            "upi",
            "net_banking"
        ],
        key="input_payment_method"
    )

with c2:

    customer_type = st.selectbox(
        "Customer Type",
        [
            "new",
            "regular",
            "loyal"
        ],
        key="input_customer_type"
    )

    attempts = st.number_input(
        "Attempts",
        min_value=1,
        value=2,
        step=1,
        key="input_attempts"
    )

    previous_success_rate = st.slider(
        "Previous Success Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
        key="input_success_rate"
    )

with c3:

    previous_failed_payments = st.number_input(
        "Previous Failed Payments",
        min_value=0,
        value=2,
        step=1,
        key="input_failed_payments"
    )

    customer_tenure_months = st.number_input(
        "Customer Tenure (Months)",
        min_value=0,
        value=5,
        step=1,
        key="input_tenure"
    )

    hours_since_failure = st.number_input(
        "Hours Since Failure",
        min_value=0,
        value=5,
        step=1,
        key="input_hours"
    )

transaction_input = pd.DataFrame([{
    "amount": amount,
    "failure_reason": failure_reason,
    "payment_method": payment_method,
    "customer_type": customer_type,
    "attempts": attempts,
    "previous_success_rate": previous_success_rate,
    "previous_failed_payments": previous_failed_payments,
    "customer_tenure_months": customer_tenure_months,
    "hours_since_failure": hours_since_failure
}])

st.divider()

if st.button(
    "Predict Recovery",
    type="primary",
    use_container_width=True,
    key="predict_button"
):

    probability = model.predict_proba(transaction_input)[0][1]
    prediction = model.predict(transaction_input)[0]

    if probability >= 0.70:
        priority = "HIGH"
        action = "Retry payment immediately"
    elif probability >= 0.40:
        priority = "MEDIUM"
        action = "Send recovery reminder and retry with an alternative payment method"
    else:
        priority = "LOW"
        action = "Avoid repeated retries and use an alternative recovery strategy"

    expected_revenue = amount * probability
    revenue_at_risk = amount - expected_revenue

    st.header("AI Recovery Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Recovery Probability",
            f"{probability * 100:.2f}%"
        )

    with c2:
        st.metric(
            "Expected Recovered Revenue",
            f"₹{expected_revenue:,.2f}"
        )

    with c3:
        st.metric(
            "Revenue at Risk",
            f"₹{revenue_at_risk:,.2f}"
        )

    st.progress(float(probability))

    if priority == "HIGH":
        st.success(f"Priority: {priority}")
    elif priority == "MEDIUM":
        st.warning(f"Priority: {priority}")
    else:
        st.error(f"Priority: {priority}")

    st.info(
        f"Recommended Recovery Action: {action}"
    )

    if prediction == 1:
        st.success(
            "AI Prediction: Payment is likely to be recovered."
        )
    else:
        st.warning(
            "AI Prediction: Payment is unlikely to be recovered."
        )

st.divider()

st.header("Revenue Recovery Analytics")

total_transactions = len(df)
total_value = df["amount"].sum()
recovered_value = df.loc[df["recovered"] == 1, "amount"].sum()
recovery_rate = df["recovered"].mean()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

with c2:
    st.metric(
        "Transaction Value",
        f"₹{total_value:,.0f}"
    )

with c3:
    st.metric(
        "Recovered Revenue",
        f"₹{recovered_value:,.0f}"
    )

with c4:
    st.metric(
        "Recovery Rate",
        f"{recovery_rate * 100:.2f}%"
    )

st.divider()

c1, c2 = st.columns(2)

with c1:
    st.subheader("Recovery by Failure Reason")

    failure_analysis = (
        df.groupby("failure_reason")["recovered"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    st.bar_chart(failure_analysis)

with c2:
    st.subheader("Recovery by Payment Method")

    payment_analysis = (
        df.groupby("payment_method")["recovered"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    st.bar_chart(payment_analysis)

c1, c2 = st.columns(2)

with c1:
    st.subheader("Recovery by Customer Type")

    customer_analysis = (
        df.groupby("customer_type")["recovered"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    st.bar_chart(customer_analysis)

with c2:
    st.subheader("Recovery by Attempts")

    attempts_analysis = (
        df.groupby("attempts")["recovered"]
        .mean()
        .mul(100)
        .sort_index()
    )

    st.line_chart(attempts_analysis)

st.divider()

st.header("Model Performance")

c1, c2 = st.columns(2)

with c1:
    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with c2:
    st.metric(
        "ROC-AUC",
        f"{roc_auc:.3f}"
    )

st.divider()

st.caption(
    "AI Revenue Recovery Engine | ML-powered payment recovery"
)