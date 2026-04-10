import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# =========================================================
# 1) Load dataset
# =========================================================
def load_adult_dataset():
    adult = fetch_openml(name="adult", version=2, as_frame=True)

    X = adult.data.copy()
    y = adult.target.copy()

    # Convert target to binary
    y = y.map({"<=50K": 0, ">50K": 1})

    return X, y


# =========================================================
# 2) Build preprocessing
# =========================================================
def build_preprocessor(X):
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor


# =========================================================
# 3) Evaluation function
# =========================================================
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("=" * 70)
    print(f"MODEL: {name}")
    print("=" * 70)
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    results = {
        "model": name,
        "accuracy": acc,
        "precision_class_1": report["1"]["precision"],
        "recall_class_1": report["1"]["recall"],
        "f1_class_1": report["1"]["f1-score"]
    }

    return results


# =========================================================
# 4) Main
# =========================================================
def main():
    # Load data
    X, y = load_adult_dataset()

    print("Dataset shape:", X.shape)
    print("\nTarget distribution:")
    print(y.value_counts(normalize=True))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Preprocessor
    preprocessor = build_preprocessor(X)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    all_results = []

    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        results = evaluate_model(name, pipeline, X_test, y_test)
        all_results.append(results)

    # Final comparison table
    results_df = pd.DataFrame(all_results).sort_values(by="accuracy", ascending=False)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()