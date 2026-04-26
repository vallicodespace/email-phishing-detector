"""Phishing email detection PoC.

Implements a lightweight supervised ML pipeline aligned with the project PoC document:
- Text preprocessing (lowercasing, punctuation removal; optional stopwords + lemmatization)
- Feature extraction using TF-IDF (word + bigram)
- Simple structural features (URL count, suspicious keyword flag, length, optional sender-domain heuristics)
- Model training and evaluation for: Logistic Regression, Linear SVM, Random Forest

Run with `--help` to see required arguments.
"""

import argparse
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Optional: NLTK stopwords/lemmatization if available.
try:
    import nltk  # type: ignore
    from nltk.corpus import stopwords  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore

    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False


LOGGER = logging.getLogger("phishing_email_poc")

URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
NON_WORD_REGEX = re.compile(r"[^a-z0-9\s]", flags=re.IGNORECASE)
MULTI_SPACE = re.compile(r"\s+")


SUSPICIOUS_KEYWORDS = (
    "verify",
    "urgent",
    "password",
    "immediately",
    "account",
    "confirm",
    "bank",
    "security",
    "login",
    "click",
    "update",
    "suspend",
)


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion_matrix: np.ndarray
    classification_report: str


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        preview = list(df.columns)[:80]
        raise ValueError(f"Missing required column(s): {missing}. Available (preview): {preview}")


def coerce_target_to_binary(y: pd.Series) -> pd.Series:
    """Coerce a 2-class label series to {0,1}.

    Supports common Kaggle encodings: {0,1}, {-1,1}, {"phishing","legitimate"}, booleans.
    """

    if y.dtype == bool:
        return y.astype(int)

    if pd.api.types.is_numeric_dtype(y):
        uniq = sorted(pd.unique(y.dropna()))
        if len(uniq) == 2:
            mapping = {uniq[0]: 0, uniq[1]: 1}
            return y.map(mapping).astype(int)

    y_str = y.astype(str).str.strip().str.lower()
    phishing_tokens = {"phishing", "malicious", "spam", "bad", "1", "true", "yes"}
    legit_tokens = {"legitimate", "benign", "ham", "good", "0", "false", "no"}

    uniq = set(pd.unique(y_str.dropna()))
    if uniq.issubset(phishing_tokens.union(legit_tokens)):
        return y_str.map(lambda v: 1 if v in phishing_tokens else 0).astype(int)

    # Last resort: parse numerics from strings.
    y_num = pd.to_numeric(y_str, errors="coerce")
    uniq2 = sorted(pd.unique(y_num.dropna()))
    if len(uniq2) == 2:
        mapping = {uniq2[0]: 0, uniq2[1]: 1}
        return y_num.map(mapping).astype(int)

    raise ValueError("Target is not clearly binary. Provide a binary label column.")


def basic_text_preprocess(text: str, *, use_nltk: bool = True) -> str:
    """Basic email text cleanup.

    Always applied:
    - lowercase
    - URL normalization
    - remove punctuation/special characters
    - collapse whitespace

    If NLTK is available and enabled:
    - stopword removal
    - lemmatization

    If NLTK resources aren't available (or download fails), the function falls back to the basic cleanup.
    """

    if text is None:
        return ""

    s = str(text).lower()
    s = URL_REGEX.sub(" URL ", s)
    s = NON_WORD_REGEX.sub(" ", s)
    s = MULTI_SPACE.sub(" ", s).strip()

    if not use_nltk or not _NLTK_AVAILABLE:
        return s

    # Best-effort downloads; offline environments should still work with basic cleanup.
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        try:
            nltk.download("stopwords", quiet=True)
        except Exception:
            return s

    try:
        nltk.data.find("corpora/wordnet")
    except Exception:
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            return s

    stops = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    tokens = [t for t in s.split() if t and t not in stops]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)


def build_structural_features(df: pd.DataFrame, text_col: str, sender_col: Optional[str]) -> pd.DataFrame:
    """Generate simple numeric features mentioned in the PoC doc."""

    text = df[text_col].fillna("").astype(str)

    url_count = text.apply(lambda s: len(URL_REGEX.findall(s))).astype(int)
    length = text.str.len().astype(int)
    suspicious_present = text.str.lower().apply(lambda s: int(any(k in s for k in SUSPICIOUS_KEYWORDS))).astype(int)

    features = pd.DataFrame(
        {
            "url_count": url_count,
            "email_length": length,
            "has_suspicious_keywords": suspicious_present,
        }
    )

    if sender_col and sender_col in df.columns:
        sender = df[sender_col].fillna("").astype(str).str.lower()

        # Handles typical forms: "Name <user@domain.com>" or "user@domain.com" or domain in a field.
        domain = sender.str.extract(r"@([^>\s]+)")[0].fillna(sender)
        domain = domain.str.replace(r"^mailto:", "", regex=True)

        features["sender_domain_length"] = domain.str.len().astype(int)
        features["sender_has_digits"] = domain.str.contains(r"\d", regex=True).astype(int)
        features["sender_domain_dot_count"] = domain.str.count(r"\.").astype(int)
    else:
        # Keep a stable schema when sender column is missing.
        features["sender_domain_length"] = 0
        features["sender_has_digits"] = 0
        features["sender_domain_dot_count"] = 0

    return features


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Metrics:
    y_pred = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    auc: Optional[float] = None
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_score))
    except Exception:
        auc = None

    return Metrics(
        accuracy=float(acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=auc,
        confusion_matrix=cm,
        classification_report=classification_report(y_test, y_pred, zero_division=0),
    )


def _make_feature_union(text_col: str, sender_col: Optional[str], use_nltk: bool) -> ColumnTransformer:
    def preprocess_text_col(X_in: pd.DataFrame) -> np.ndarray:
        col = X_in.iloc[:, 0].astype(str).fillna("")
        return col.apply(lambda s: basic_text_preprocess(s, use_nltk=use_nltk)).to_numpy()

    text_cleaner = FunctionTransformer(preprocess_text_col, validate=False)

    text_pipeline = Pipeline(
        steps=[
            ("clean", text_cleaner),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    def structural_features_transform(X_in: pd.DataFrame) -> pd.DataFrame:
        return build_structural_features(X_in, text_col=text_col, sender_col=sender_col)

    structural_transformer = FunctionTransformer(structural_features_transform, validate=False)

    feature_input_cols = [text_col]
    if sender_col:
        feature_input_cols.append(sender_col)

    return ColumnTransformer(
        transformers=[
            ("text", text_pipeline, [text_col]),
            (
                "struct",
                Pipeline(
                    steps=[
                        ("struct_features", structural_transformer),
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                feature_input_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def _to_dense(X_mat):
    return X_mat.toarray() if hasattr(X_mat, "toarray") else X_mat


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple Phishing Email Detection POC (LR/SVM/RF).")
    ap.add_argument("--csv", required=True, help="Path to input CSV (Kaggle or local).")
    ap.add_argument("--target", required=True, help="Target/label column (phishing vs legitimate).")
    ap.add_argument("--text-col", required=True, help="Email text column (body/content).")
    ap.add_argument("--sender-col", default=None, help="Optional sender column (e.g., from/email).")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--artifacts", default="artifacts", help="Folder to save trained model pipelines.")
    ap.add_argument("--no-nltk", action="store_true", help="Disable NLTK stopwords/lemmatization.")
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = ap.parse_args()

    _configure_logging(args.verbose)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv).dropna(axis=1, how="all").drop_duplicates()

    required = [args.target, args.text_col]
    if args.sender_col:
        required.append(args.sender_col)
    _ensure_columns(df, required)

    y = coerce_target_to_binary(df[args.target])

    feature_cols = [args.text_col] + ([args.sender_col] if args.sender_col else [])
    X = df[feature_cols].copy()

    # Ensure deterministic split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    use_nltk = not args.no_nltk
    if use_nltk and not _NLTK_AVAILABLE:
        LOGGER.info("NLTK not available; using basic regex-based preprocessing.")

    features = _make_feature_union(text_col=args.text_col, sender_col=args.sender_col, use_nltk=use_nltk)

    models: Dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                ("features", features),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "svm_linear": Pipeline(
            steps=[
                ("features", features),
                ("model", LinearSVC(class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("features", features),
                ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        random_state=args.random_state,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
    }

    os.makedirs(args.artifacts, exist_ok=True)

    scores: List[Tuple[str, float]] = []

    for name, pipe in models.items():
        LOGGER.info("Training model: %s", name)
        pipe.fit(X_train, y_train)

        metrics = evaluate_model(pipe, X_test, y_test)
        scores.append((name, metrics.f1))

        out_path = os.path.join(args.artifacts, f"{name}.joblib")
        joblib.dump(pipe, out_path)

        print("\n" + "=" * 80)
        print(f"MODEL: {name}")
        print(f"SAVED: {out_path}")
        print(
            f"accuracy={metrics.accuracy:.4f} precision={metrics.precision:.4f} "
            f"recall={metrics.recall:.4f} f1={metrics.f1:.4f} roc_auc={metrics.roc_auc}"
        )
        print("Confusion matrix [[TN FP],[FN TP]]:")
        print(metrics.confusion_matrix)
        print(metrics.classification_report)

    best = max(scores, key=lambda t: t[1])
    print("\n" + "=" * 80)
    print(f"Best by F1: {best[0]} (F1={best[1]:.4f})")


if __name__ == "__main__":
    main()