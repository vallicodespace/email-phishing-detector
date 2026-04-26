import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# LOAD DATA
# -----------------------------
csv_path = "data/PhishingEmailData.csv"

try:
    df = pd.read_csv(csv_path, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding="latin-1")

df.columns = df.columns.str.strip()

TEXT_COL = "Email_Content"

print("Shape:", df.shape)


# -----------------------------
# BASIC CLEANING + FEATURES
# -----------------------------
URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

KEYWORDS = [
    "verify", "urgent", "password", "account",
    "login", "bank", "click", "update",
    "suspend", "confirm", "security"
]


def clean_text(X):
    return X.iloc[:, 0].str.lower().to_numpy()


def struct_features(df):
    text = df[TEXT_COL].astype(str)

    return pd.DataFrame({
        "length": text.str.len(),
        "url_count": text.apply(lambda x: len(URL_REGEX.findall(x))),
        "keyword_hits": text.apply(lambda x: sum(k in x.lower() for k in KEYWORDS))
    })


# -----------------------------
# WEAK LABEL (for training only)
# -----------------------------
def make_label(row):
    text = str(row[TEXT_COL]).lower()

    score = 0
    score += len(URL_REGEX.findall(text)) * 2
    score += sum(k in text for k in KEYWORDS)
    if len(text) < 80:
        score += 1

    return 1 if score >= 3 else 0


df["label"] = df.apply(make_label, axis=1)

print("\nLabel distribution:")
print(df["label"].value_counts())


# -----------------------------
# SPLIT
# -----------------------------
X = df[[TEXT_COL]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# PIPELINE FEATURES
# -----------------------------
features = ColumnTransformer([
    ("text", Pipeline([
        ("clean", FunctionTransformer(clean_text, validate=False)),
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2))
    ]), [TEXT_COL]),

    ("struct", Pipeline([
        ("feat", FunctionTransformer(lambda X: struct_features(X), validate=False)),
        ("imp", SimpleImputer(strategy="median"))
    ]), [TEXT_COL])
])


# -----------------------------
# MODELS (REQUIRED 3)
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVM": LinearSVC(max_iter=5000),
    "Random Forest": RandomForestClassifier(n_estimators=200)
}


trained_models = {}

print("\n🔧 Training models...\n")

for name, model in models.items():

    pipe = Pipeline([
        ("features", features),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    print("=" * 60)
    print(name)
    print("Accuracy:", round(accuracy_score(y_test, pred), 3))
    print("F1:", round(f1_score(y_test, pred), 3))

    trained_models[name] = pipe


# -----------------------------
# RISK FUNCTION (INDEPENDENT)
# -----------------------------
def risk_level(text):
    text = str(text).lower()

    score = 0
    score += len(URL_REGEX.findall(text)) * 2
    score += sum(k in text for k in KEYWORDS)
    if len(text) < 80:
        score += 1

    return "HIGH RISK" if score >= 3 else "LOW RISK"


# -----------------------------
# FINAL OUTPUT (BOTH INFO SEPARATE)
# -----------------------------
print("\n================ EMAIL RESULTS ================\n")

sample = df[[TEXT_COL]].head(10)

for i, email in enumerate(sample[TEXT_COL]):

    print("-" * 70)
    print("📩 EMAIL:", email[:120], "...")

    # 🔹 Model predictions (all 3)
    for name, model in trained_models.items():
        pred = model.predict(sample.iloc[[i]])[0]
        label = "PHISHING" if pred == 1 else "LEGITIMATE"
        print(f"🤖 {name}: {label}")

    # 🔹 Risk level (separate system)
    print("🚨 RISK LEVEL:", risk_level(email))