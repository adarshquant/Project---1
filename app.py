import pandas as pd
import random

random.seed(42)

n = 5000
data = []

failure_reasons = [
    "network_error",
    "insufficient_funds",
    "bank_decline",
    "card_expired",
    "technical_error"
]

payment_methods = [
    "credit_card",
    "debit_card",
    "upi",
    "net_banking"
]

customer_types = [
    "new",
    "regular",
    "loyal"
]

for i in range(n):

    amount = random.randint(200, 15000)

    failure_reason = random.choice(failure_reasons)

    payment_method = random.choice(payment_methods)

    customer_type = random.choice(customer_types)

    attempts = random.randint(1, 3)

    previous_success_rate = round(random.uniform(0.30, 0.98), 2)

    previous_failed_payments = random.randint(0, 6)

    customer_tenure_months = random.randint(1, 60)

    hours_since_failure = random.randint(1, 72)


    recovery_probability = 0.50


    if failure_reason == "network_error":
        recovery_probability += 0.20

    elif failure_reason == "technical_error":
        recovery_probability += 0.15

    elif failure_reason == "insufficient_funds":
        recovery_probability -= 0.10

    elif failure_reason == "bank_decline":
        recovery_probability -= 0.15

    elif failure_reason == "card_expired":
        recovery_probability -= 0.20
    recovery_probability += (previous_success_rate - 0.50) * 0.50


    recovery_probability -= previous_failed_payments * 0.04


    recovery_probability -= (attempts - 1) * 0.08


    if customer_type == "loyal":
        recovery_probability += 0.10

    elif customer_type == "new":
        recovery_probability -= 0.05

    
    recovery_probability = max(
        0.05,
        min(0.95, recovery_probability)
    )

    recovered = int(
        random.random() < recovery_probability
    )

    data.append({
        "transaction_id": f"TXN{i+1:05d}",
        "amount": amount,
        "failure_reason": failure_reason,
        "payment_method": payment_method,
        "customer_type": customer_type,
        "attempts": attempts,
        "previous_success_rate": previous_success_rate,
        "previous_failed_payments": previous_failed_payments,
        "customer_tenure_months": customer_tenure_months,
        "hours_since_failure": hours_since_failure,
        "recovered": recovered
    })


df = pd.DataFrame(data)

df.to_csv(
    "data/transactions.csv",
    index=False
)

print("Dataset created successfully!")
print(f"Total transactions: {len(df)}")
print()
print(df.head())
print()
print("Recovery rate:")
print(df["recovered"].mean())
print("\n--- DATASET INFORMATION ---")
print(df.info())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- RECOVERY COUNTS ---")
print(df["recovered"].value_counts())

print("\n--- RECOVERY RATE BY FAILURE REASON ---")
print(
    df.groupby("failure_reason")["recovered"]
    .mean()
    .sort_values(ascending=False)
)

print("\n--- RECOVERY RATE BY CUSTOMER TYPE ---")
print(
    df.groupby("customer_type")["recovered"]
    .mean()
    .sort_values(ascending=False)
)
import matplotlib.pyplot as plt


recovery_by_reason = (
    df.groupby("failure_reason")["recovered"]
    .mean()
    .sort_values(ascending=False)
)

print("\nRecovery rate by failure reason:")
print(recovery_by_reason)

recovery_by_reason.plot(
    kind="bar",
    title="Recovery Rate by Failure Reason",
    ylabel="Recovery Rate",
    xlabel="Failure Reason"
)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
recovery_by_customer = (
    df.groupby("customer_type")["recovered"]
    .mean()
    .sort_values(ascending=False)
)

print(recovery_by_customer)

recovery_by_customer.plot(
    kind="bar",
    title="Recovery Rate by Customer Type",
    ylabel="Recovery Rate",
    xlabel="Customer Type"
)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
recovery_by_attempts = (
    df.groupby("attempts")["recovered"]
    .mean()
)

print(recovery_by_attempts)

recovery_by_attempts.plot(
    kind="bar",
    title="Recovery Rate by Number of Attempts",
    ylabel="Recovery Rate",
    xlabel="Attempts"
)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nAverage amount by recovery:")
print(df.groupby("recovered")["amount"].mean())

df.boxplot(
    column="amount",
    by="recovered"
)

plt.title("Transaction Amount vs Recovery")
plt.suptitle("")
plt.xlabel("Recovered")
plt.ylabel("Amount")
plt.tight_layout()
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]
    print("\nROC-AUC Score:")
    print(round(roc_auc_score(y_test, y_prob), 3))

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.show()
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

y_prob = model.predict_proba(X_test)[:, 1]

print("\nROC-AUC Score:")
print(round(roc_auc_score(y_test, y_prob), 3))

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve")
plt.tight_layout()
plt.show()
sample_transaction = X_test.iloc[[0]]

sample_prediction = model.predict(sample_transaction)

sample_probability = model.predict_proba(sample_transaction)[0][1]

print("\nSample Transaction Prediction:")
print("Recovered:", sample_prediction[0])
print("Recovery Probability:", sample_probability)
print("\nEnter Transaction Details:")

transaction_input = {}

for feature in X.columns:
    if feature in categorical_features:
        transaction_input[feature] = input(f"Enter {feature}: ")
    else:
        transaction_input[feature] = float(input(f"Enter {feature}: "))

new_transaction = pd.DataFrame([transaction_input])

prediction = model.predict(new_transaction)[0]
probability = model.predict_proba(new_transaction)[0][1]

print("\nPrediction:")
print("Recovered:", prediction)
print("Recovery Probability:", probability)
if probability >= 0.70:
    priority = "HIGH"
    action = "Retry payment immediately"
elif probability >= 0.40:
    priority = "MEDIUM"
    action = "Send recovery reminder and retry with an alternative payment method"
else:
    priority = "LOW"
    action = "Avoid repeated retries and use an alternative recovery strategy"

expected_revenue = transaction_input["amount"] * probability
revenue_at_risk = transaction_input["amount"] - expected_revenue

print("\nRecovery Strategy:")
print("Priority:", priority)
print("Recommended Action:", action)

print("\nRevenue Analysis:")
print("Transaction Amount: ₹", round(transaction_input["amount"], 2))
print("Recovery Probability:", round(probability * 100, 2), "%")
print("Expected Recovered Revenue: ₹", round(expected_revenue, 2))
print("Potential Revenue at Risk: ₹", round(revenue_at_risk, 2))

print("\nAI Recovery Decision:")
if priority == "HIGH":
    print("High probability of recovery. Prioritize this transaction.")
elif priority == "MEDIUM":
    print("Moderate recovery probability. Use a targeted recovery strategy.")
else:
    print("Low recovery probability. Avoid unnecessary retry attempts.")