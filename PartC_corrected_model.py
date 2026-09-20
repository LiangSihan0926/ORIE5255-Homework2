#!/usr/bin/env python3
"""ORIE 5255 Homework 2 - independent model-risk benchmark.

Reproduces the submitted development script (week3_4_bad_model.py) end to end,
then rebuilds the validation design on an application-time feature set with a
chronological development / validation / final out-of-time split.

The final out-of-time test is scored exactly once, after the model family,
hyperparameters and decision threshold have been frozen on development and
validation evidence only.

Usage:
    python Liang_corrected_model.py --data week3_4_credit_model_data_1000.csv

Outputs (written to --output-dir, default: the script directory):
    Liang_results.csv                 bad-versus-corrected metric comparison
    Liang_data_audit.csv              Part B audit findings
    Liang_data_dictionary.csv         field-level timing classification
    Liang_cv_results.csv              temporal cross-validation grid
    Liang_subgroup_analysis.csv       age-band subgroup performance
    Liang_temporal_analysis.csv       quarterly label maturity and event rate
    Liang_final_test_predictions.csv  record-level out-of-time scores
    Liang_bad_model_top_features.csv  mutual-information screen of the bad model
    Liang_calibration_summary.csv     E/O ratio, calibration intercept and slope
    Liang_near_duplicate_screen.csv   exact and near-duplicate screen
    Liang_outlier_screen.csv          per-variable range and rule screen
    Liang_subgroup_extended.csv       home ownership, channel and state subgroups
    Liang_challenger_benchmarks.csv   bureau-score challenger and exclusion sensitivity
    Liang_leakage_attribution.csv     Part D one-fault-at-a-time attribution study
    Liang_run_manifest.json           seeds, input hash, library versions
    four supporting figures
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "default_12m"
BAD_SEED = 7
CORRECTED_SEED = 42

# Application-time features only.
# Age, state and zip_prefix are kept for subgroup testing but not used for fitting.
# employment_years is dropped until the 55 implausible values are fixed at source.
MODEL_NUMERIC = [
    "annual_income_at_application",
    "credit_score",
    "debt_to_income",
    "loan_amount",
    "term_months",
    "num_open_accounts",
    "delinq_2y",
    "inq_6m",
    "credit_utilization",
    "prior_defaults",
    "bankruptcy_flag",
    "unemployment_rate",
    "prime_rate",
]
MODEL_CATEGORICAL = ["home_ownership", "purpose", "channel"]
MODEL_FEATURES = MODEL_NUMERIC + MODEL_CATEGORICAL


FIELD_CLASSIFICATION = {
    "customer_id": ("IDENTIFIER", "Persistent identifier. Use for traceability and grouping, never as a predictor."),
    "application_date": ("KNOWN_AT_DECISION_TIME", "Prediction timestamp. Use for eligibility and chronological splitting, not as a numeric predictor."),
    "age": ("KNOWN_AT_DECISION_TIME; POTENTIAL_PROXY", "Protected-age and fair-lending concern. Excluded from fitting and retained for subgroup testing."),
    "state": ("KNOWN_AT_DECISION_TIME; POTENTIAL_PROXY", "Geography can proxy protected characteristics. Excluded pending fair-lending review."),
    "zip_prefix": ("KNOWN_AT_DECISION_TIME; POTENTIAL_PROXY", "High-cardinality geography and redlining/proxy concern. Excluded from fitting."),
    "annual_income_at_application": ("KNOWN_AT_DECISION_TIME", "Reported application income. Retained with training-only imputation and a missingness indicator."),
    "current_income": ("POST_DECISION; UNCLEAR", "The extraction date and as-of convention are not documented. Excluded."),
    "employment_years": ("KNOWN_AT_DECISION_TIME; UNCLEAR", "Application-time in concept, but 55 values exceed age minus 14. Excluded until reconciled."),
    "credit_score": ("KNOWN_AT_DECISION_TIME", "Bureau score at application. Retained with training-only imputation."),
    "debt_to_income": ("KNOWN_AT_DECISION_TIME", "Application affordability measure. Retained."),
    "loan_amount": ("KNOWN_AT_DECISION_TIME", "Requested/originated amount. Retained for this benchmark."),
    "interest_rate": ("UNCLEAR", "May be assigned after underwriting rather than known at the stated application-time prediction point. Excluded."),
    "term_months": ("KNOWN_AT_DECISION_TIME", "Requested loan term. Retained."),
    "home_ownership": ("KNOWN_AT_DECISION_TIME; POTENTIAL_PROXY", "Application field. Retained for the benchmark but requires proxy and subgroup review."),
    "purpose": ("KNOWN_AT_DECISION_TIME", "Stated loan purpose. Retained."),
    "channel": ("KNOWN_AT_DECISION_TIME; POTENTIAL_PROXY", "Application channel. Retained provisionally and monitored for access-related proxy effects."),
    "num_open_accounts": ("KNOWN_AT_DECISION_TIME", "Bureau history available at application. Retained."),
    "delinq_2y": ("KNOWN_AT_DECISION_TIME", "Historical delinquency count. Retained."),
    "inq_6m": ("KNOWN_AT_DECISION_TIME", "Historical credit inquiries. Retained."),
    "credit_utilization": ("KNOWN_AT_DECISION_TIME", "Bureau utilization at application. Retained with training-only imputation."),
    "prior_defaults": ("KNOWN_AT_DECISION_TIME", "Prior outcome history, not the future target. Retained subject to point-in-time sourcing."),
    "bankruptcy_flag": ("KNOWN_AT_DECISION_TIME", "Application-time public-record indicator. Retained subject to legal and policy review."),
    "unemployment_rate": ("KNOWN_AT_DECISION_TIME; MACRO", "Macro value at application. Retained; vintage and publication lag must be controlled in production."),
    "prime_rate": ("KNOWN_AT_DECISION_TIME; MACRO", "Macro rate at application. Retained; vintage must be point-in-time reproducible."),
    "manual_review_flag": ("POST_DECISION; UNCLEAR; POTENTIAL_PROXY", "May reflect downstream process or human discretion. Excluded."),
    "underwriter_score": ("POST_DECISION; UNCLEAR; POTENTIAL_PROXY", "Likely an underwriting output rather than raw application information. Excluded."),
    "model_version": ("IDENTIFIER; UNCLEAR", "System/process metadata strongly aligned with time. Excluded from fitting."),
    "months_observed": ("POST_DECISION; OUTCOME_DERIVED", "Measures follow-up length. Use only to establish label maturity."),
    "label_available": ("POST_DECISION; OUTCOME_DERIVED", "Indicates target observability. Use only for eligibility."),
    "days_past_due_6m_after": ("POST_DECISION; OUTCOME_DERIVED", "Occurs after application and directly reveals repayment behavior. Excluded."),
    "collections_contacted_9m_after": ("POST_DECISION; OUTCOME_DERIVED", "Post-application collections action. Excluded."),
    "charged_off_amount_12m": ("POST_DECISION; OUTCOME_DERIVED", "Direct future charge-off outcome. Excluded."),
    "account_status_12m": ("POST_DECISION; OUTCOME_DERIVED", "Twelve-month status nearly defines the target. Excluded."),
    "current_balance": ("POST_DECISION; UNCLEAR", "Balance as of an undocumented later extraction date. Excluded."),
    "default_12m": ("OUTCOME_DERIVED", "Prediction target. Never included in predictors."),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="Path to the supplied raw CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument(
        "--attribution-seeds",
        type=int,
        default=5,
        help="Seeds used by the Part D leakage-attribution study.",
    )
    return parser.parse_args()


def resolve_data_path(candidate):
    if candidate is not None:
        path = candidate.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    # The supplied CSV is expected beside this script, or given with --data.
    script_dir = Path(__file__).resolve().parent
    default_path = script_dir / "week3_4_credit_model_data_1000.csv"
    if default_path.exists():
        return default_path.resolve()
    raise FileNotFoundError(
        "week3_4_credit_model_data_1000.csv was not found next to this script. "
        "Supply its location with --data."
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_auc(y_true, probability):
    y_array = np.asarray(y_true, dtype=int)
    if np.unique(y_array).size < 2:
        return float("nan")
    return float(roc_auc_score(y_array, probability))


def evaluate_binary(y_true, probability, threshold):
    y_array = np.asarray(y_true, dtype=int)
    prediction = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_array, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": safe_auc(y_array, probability),
        "pr_auc": float(average_precision_score(y_array, probability)),
        "accuracy": float(accuracy_score(y_array, prediction)),
        "precision": float(precision_score(y_array, prediction, zero_division=0)),
        "recall": float(recall_score(y_array, prediction, zero_division=0)),
        "f1": float(f1_score(y_array, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y_array, probability)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "confusion_matrix": f"[[{tn}, {fp}], [{fn}, {tp}]]",
    }


def reproduce_bad_model(raw):
    """Reproduce the submitted procedure, including its contamination."""
    df = raw.copy()
    df[TARGET] = df[TARGET].fillna(0).astype(int)
    df["customer_number"] = df["customer_id"].str.replace("CUST-", "", regex=False).astype(int)
    df["application_year"] = df["application_date"].dt.year
    df["application_month"] = df["application_date"].dt.month
    df["application_day"] = df["application_date"].dt.day
    df["application_dayofweek"] = df["application_date"].dt.dayofweek
    df["loan_to_income"] = df["loan_amount"] / df["annual_income_at_application"].replace(0, np.nan)
    df["balance_to_loan"] = df["current_balance"] / df["loan_amount"].replace(0, np.nan)
    df["income_change"] = (
        (df["current_income"] - df["annual_income_at_application"])
        / df["annual_income_at_application"].replace(0, np.nan)
    )
    df = df.drop(columns=["application_date", "customer_id"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(TARGET)
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    df[numeric_cols] = SimpleImputer(strategy="median").fit_transform(df[numeric_cols])
    if categorical_cols:
        df[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(
            df[categorical_cols]
        )
    for col in categorical_cols:
        full_sample_target_rate = df.groupby(col)[TARGET].mean()
        df[f"{col}_target_rate"] = df[col].map(full_sample_target_rate)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True)).fillna(0)

    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    x_scaled = pd.DataFrame(
        StandardScaler().fit_transform(x), columns=x.columns, index=x.index
    )

    rng = np.random.RandomState(BAD_SEED)
    default_idx = y[y == 1].index.to_numpy()
    nondefault_idx = y[y == 0].index.to_numpy()
    extra_default_idx = rng.choice(
        default_idx, size=len(nondefault_idx) - len(default_idx), replace=True
    )
    balanced_idx = np.concatenate([nondefault_idx, default_idx, extra_default_idx])
    rng.shuffle(balanced_idx)
    x_balanced = x_scaled.loc[balanced_idx].reset_index(drop=True)
    y_balanced = y.loc[balanced_idx].reset_index(drop=True)
    source_rows = pd.Series(balanced_idx, name="source_row").reset_index(drop=True)

    mutual_information = mutual_info_classif(
        x_balanced, y_balanced, random_state=BAD_SEED
    )
    mi_table = pd.DataFrame(
        {"feature": x_balanced.columns, "mutual_information": mutual_information}
    ).sort_values("mutual_information", ascending=False)
    selected_features = mi_table.head(35)["feature"].tolist()
    x_selected = x_balanced[selected_features]

    x_train, x_test, y_train, y_test, source_train, source_test = train_test_split(
        x_selected,
        y_balanced,
        source_rows,
        test_size=0.25,
        random_state=BAD_SEED,
        shuffle=True,
    )
    models = {
        "logistic": LogisticRegression(
            C=1.0, max_iter=5000, class_weight=None, random_state=BAD_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=BAD_SEED,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.04, max_depth=3, random_state=BAD_SEED
        ),
    }
    comparison = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        comparison.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, prediction),
                "binary_auc": roc_auc_score(y_test, prediction),
            }
        )
    comparison_df = pd.DataFrame(comparison).sort_values(
        ["accuracy", "binary_auc"], ascending=False
    )
    champion_name = str(comparison_df.iloc[0]["model"])

    tuned = []
    if champion_name == "random_forest":
        for depth in [3, 5, 8, 12, None]:
            model = RandomForestClassifier(
                n_estimators=700,
                max_depth=depth,
                min_samples_leaf=1,
                random_state=BAD_SEED,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            tuned.append((accuracy_score(y_test, model.predict(x_test)), depth, model))
    elif champion_name == "logistic":
        for c_value in [0.01, 0.1, 1, 10, 100]:
            model = LogisticRegression(
                C=c_value, max_iter=5000, random_state=BAD_SEED
            )
            model.fit(x_train, y_train)
            tuned.append((accuracy_score(y_test, model.predict(x_test)), c_value, model))
    else:
        for depth in [1, 2, 3, 4]:
            model = GradientBoostingClassifier(
                n_estimators=350,
                learning_rate=0.04,
                max_depth=depth,
                random_state=BAD_SEED,
            )
            model.fit(x_train, y_train)
            tuned.append((accuracy_score(y_test, model.predict(x_test)), depth, model))
    tuned.sort(key=lambda item: item[0], reverse=True)
    _, tuned_parameter, champion = tuned[0]
    test_probability = champion.predict_proba(x_test)[:, 1]

    threshold_rows = []
    for threshold in np.arange(0.05, 0.96, 0.01):
        prediction = (test_probability >= threshold).astype(int)
        threshold_rows.append(
            (
                threshold,
                accuracy_score(y_test, prediction),
                f1_score(y_test, prediction, zero_division=0),
            )
        )
    threshold_df = pd.DataFrame(threshold_rows, columns=["threshold", "accuracy", "f1"])
    threshold = float(
        threshold_df.sort_values(["accuracy", "f1"], ascending=False).iloc[0]["threshold"]
    )
    metrics = evaluate_binary(y_test, test_probability, threshold)
    metrics["roc_auc_reported_binary"] = float(
        roc_auc_score(y_test, (test_probability >= threshold).astype(int))
    )

    shared_sources = set(source_train.tolist()).intersection(source_test.tolist())
    contaminated_test_mask = source_test.isin(shared_sources).to_numpy()
    contamination_count = int(contaminated_test_mask.sum())
    contaminated_default_count = int(
        ((y_test.to_numpy() == 1) & contaminated_test_mask).sum()
    )
    return {
        "metrics": metrics,
        "champion": champion_name,
        "tuned_parameter": tuned_parameter,
        "threshold": threshold,
        "modeling_population_size": int(len(raw)),
        "evaluation_size": int(len(y_test)),
        "evaluation_defaults": int(y_test.sum()),
        "evaluation_prevalence": float(y_test.mean()),
        "feature_count": int(len(selected_features)),
        "selected_features": selected_features,
        "top_mi": mi_table.head(20).copy(),
        "test_contamination_count": contamination_count,
        "test_contamination_rate": float(contamination_count / len(y_test)),
        "contaminated_default_count": contaminated_default_count,
        "unique_test_source_rows": int(source_test.nunique()),
    }


def prepare_population(raw):
    # Rows that match on every field except customer_id are treated as probable
    # duplicates and one of each pair is dropped.
    signature_cols = [col for col in raw.columns if col != "customer_id"]
    duplicate_rows = int(raw.duplicated(signature_cols, keep=False).sum())
    duplicate_groups = int(
        pd.util.hash_pandas_object(raw[signature_cols], index=False)
        .loc[raw.duplicated(signature_cols, keep=False)]
        .nunique()
    )
    deduplicated = raw.drop_duplicates(signature_cols, keep="first").copy()
    eligible_mask = (
        deduplicated[TARGET].notna()
        & deduplicated["label_available"].eq(1)
        & deduplicated["months_observed"].ge(12)
    )
    eligible = deduplicated.loc[eligible_mask].copy()
    eligible[TARGET] = eligible[TARGET].astype(int)
    eligible = eligible.sort_values(["application_date", "customer_id"]).reset_index(drop=True)
    audit_counts = {
        "probable_duplicate_rows": duplicate_rows,
        "probable_duplicate_groups": duplicate_groups,
        "duplicates_removed": int(len(raw) - len(deduplicated)),
        "eligible_rows": int(len(eligible)),
        "eligible_defaults": int(eligible[TARGET].sum()),
    }
    return deduplicated, eligible, audit_counts


def chronological_split(eligible):
    development_boundary = eligible["application_date"].quantile(0.60, interpolation="higher")
    final_test_boundary = eligible["application_date"].quantile(0.80, interpolation="higher")
    development = eligible.loc[eligible["application_date"] < development_boundary].copy()
    validation = eligible.loc[
        eligible["application_date"].ge(development_boundary)
        & eligible["application_date"].lt(final_test_boundary)
    ].copy()
    final_test = eligible.loc[eligible["application_date"].ge(final_test_boundary)].copy()
    if min(development[TARGET].sum(), validation[TARGET].sum(), final_test[TARGET].sum()) <= 0:
        raise ValueError("Each chronological partition must contain at least one default.")
    if set(development.customer_id) & set(validation.customer_id):
        raise ValueError("Customer overlap between development and validation sets.")
    if set(development.customer_id) & set(final_test.customer_id):
        raise ValueError("Customer overlap between development and final test sets.")
    if set(validation.customer_id) & set(final_test.customer_id):
        raise ValueError("Customer overlap between validation and final test sets.")
    return development, validation, final_test, development_boundary, final_test_boundary


def corrected_pipeline(c_value, class_weight):
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, MODEL_NUMERIC),
            ("categorical", categorical_pipeline, MODEL_CATEGORICAL),
        ]
    )
    model = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        max_iter=5000,
        solver="liblinear",
        random_state=CORRECTED_SEED,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def tune_corrected_model(development):
    x_dev = development[MODEL_FEATURES]
    y_dev = development[TARGET]
    splitter = TimeSeriesSplit(n_splits=3)
    rows = []
    for class_weight in [None, "balanced"]:
        for c_value in [0.01, 0.1, 1.0, 10.0]:
            fold_metrics = []
            for fold, (train_idx, validation_idx) in enumerate(splitter.split(x_dev), start=1):
                y_train = y_dev.iloc[train_idx]
                y_validation = y_dev.iloc[validation_idx]
                if y_train.nunique() < 2 or y_validation.nunique() < 2:
                    raise ValueError(f"Temporal fold {fold} lacks both target classes.")
                pipeline = corrected_pipeline(c_value, class_weight)
                pipeline.fit(x_dev.iloc[train_idx], y_train)
                probability = pipeline.predict_proba(x_dev.iloc[validation_idx])[:, 1]
                fold_metrics.append(
                    {
                        "fold": fold,
                        "train_rows": len(train_idx),
                        "train_defaults": int(y_train.sum()),
                        "validation_rows": len(validation_idx),
                        "validation_defaults": int(y_validation.sum()),
                        "pr_auc": average_precision_score(y_validation, probability),
                        "roc_auc": roc_auc_score(y_validation, probability),
                        "brier_score": brier_score_loss(y_validation, probability),
                    }
                )
            rows.append(
                {
                    "C": c_value,
                    "class_weight": "None" if class_weight is None else class_weight,
                    "mean_pr_auc": np.mean([item["pr_auc"] for item in fold_metrics]),
                    "std_pr_auc": np.std([item["pr_auc"] for item in fold_metrics], ddof=1),
                    "mean_roc_auc": np.mean([item["roc_auc"] for item in fold_metrics]),
                    "mean_brier_score": np.mean([item["brier_score"] for item in fold_metrics]),
                    "fold_detail": json.dumps(fold_metrics),
                }
            )
    tuning = pd.DataFrame(rows)
    best_mean_pr = float(tuning["mean_pr_auc"].max())
    # Treat any grid point within 0.01 mean PR AUC of the best as a tie, then
    # break the tie on mean Brier score and stronger regularization. With only
    # 12 development defaults a small AP gain is mostly noise.
    candidates = tuning.loc[tuning["mean_pr_auc"] >= best_mean_pr - 0.01].copy()
    candidates = candidates.sort_values(
        ["mean_brier_score", "C"], ascending=[True, True]
    )
    selected_index = int(candidates.index[0])
    tuning["selected"] = False
    tuning.loc[selected_index, "selected"] = True
    selected_row = tuning.loc[selected_index]
    selected = {
        "C": float(selected_row["C"]),
        "class_weight": None
        if selected_row["class_weight"] == "None"
        else str(selected_row["class_weight"]),
    }
    return tuning, selected


def bootstrap_intervals(y_true, probability, repetitions):
    rng = np.random.RandomState(CORRECTED_SEED)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    values = []
    for _ in range(repetitions):
        sample = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        rng.shuffle(sample)
        sampled_y = y_true[sample]
        sampled_probability = probability[sample]
        values.append(
            (
                roc_auc_score(sampled_y, sampled_probability),
                average_precision_score(sampled_y, sampled_probability),
                brier_score_loss(sampled_y, sampled_probability),
            )
        )
    array = np.asarray(values)
    lower = np.percentile(array, 2.5, axis=0)
    upper = np.percentile(array, 97.5, axis=0)
    return {
        "roc_auc_ci_low": float(lower[0]),
        "roc_auc_ci_high": float(upper[0]),
        "pr_auc_ci_low": float(lower[1]),
        "pr_auc_ci_high": float(upper[1]),
        "brier_ci_low": float(lower[2]),
        "brier_ci_high": float(upper[2]),
    }


def corrected_model_results(development, validation, final_test, bootstrap_reps):
    tuning, selected = tune_corrected_model(development)
    pipeline = corrected_pipeline(selected["C"], selected["class_weight"])
    pipeline.fit(development[MODEL_FEATURES], development[TARGET])
    validation_probability = pipeline.predict_proba(validation[MODEL_FEATURES])[:, 1]

    # Threshold rule fixed on validation only: take the highest cutoff that still
    # catches all validation defaults. Provisional, because validation has only
    # 6 defaults and no loss matrix was supplied. The final test is not used here.
    threshold = float(validation_probability[validation[TARGET].to_numpy() == 1].min())
    final_probability = pipeline.predict_proba(final_test[MODEL_FEATURES])[:, 1]
    metrics = evaluate_binary(final_test[TARGET], final_probability, threshold)
    intervals = bootstrap_intervals(
        final_test[TARGET].to_numpy(dtype=int), final_probability, bootstrap_reps
    )
    transformed_feature_count = len(
        pipeline.named_steps["preprocess"].get_feature_names_out()
    )
    return {
        "pipeline": pipeline,
        "tuning": tuning,
        "selected": selected,
        "threshold": threshold,
        "validation_probability": validation_probability,
        "final_probability": final_probability,
        "metrics": metrics,
        "intervals": intervals,
        "transformed_feature_count": transformed_feature_count,
    }


def data_dictionary(raw):
    rows = []
    for field in raw.columns:
        classification, rationale = FIELD_CLASSIFICATION[field]
        rows.append(
            {
                "field": field,
                "classification": classification,
                "used_in_corrected_model": field in MODEL_FEATURES,
                "corrected_model_role": (
                    "Predictor"
                    if field in MODEL_FEATURES
                    else "Eligibility/split/audit only"
                    if field in {"customer_id", "application_date", "months_observed", "label_available", "age", "state"}
                    else "Excluded"
                ),
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def build_data_audit(
    raw,
    deduplicated,
    eligible,
    audit_counts,
    development,
    validation,
    final_test,
    bad_result,
):
    invalid_employment = (
        raw["employment_years"].notna()
        & raw["employment_years"].gt(raw["age"] - 14)
    )
    missing_rates = raw.isna().mean()
    observed = raw.loc[raw[TARGET].notna()].copy()
    observed["year"] = observed["application_date"].dt.year
    year_rates = observed.groupby("year")[TARGET].mean()
    unavailable_at_12 = int(
        ((raw["label_available"] == 0) & raw["months_observed"].ge(12)).sum()
    )
    default_missing_util = float(
        eligible.loc[eligible[TARGET].eq(1), "credit_utilization"].isna().mean()
    )
    nondefault_missing_util = float(
        eligible.loc[eligible[TARGET].eq(0), "credit_utilization"].isna().mean()
    )
    rows = [
        ("Raw population", len(raw), "1,000 supplied records; the raw CSV remains unchanged."),
        ("Observed targets", int(raw[TARGET].notna().sum()), "805 records have observed 12-month outcomes."),
        ("Missing targets", int(raw[TARGET].isna().sum()), "195 records (19.5%) are right-censored and cannot be labeled non-default."),
        ("Eligible after deduplication", len(eligible), f"{int(eligible[TARGET].sum())} defaults; prevalence {eligible[TARGET].mean():.2%}."),
        ("Unavailable labels at rounded 12.0 months", unavailable_at_12, "Five rows remain unavailable at a rounded 12.0 months; label_available is retained as the controlling maturity flag."),
        ("Probable duplicate rows", audit_counts["probable_duplicate_rows"], f"{audit_counts['probable_duplicate_groups']} pairs are identical on every supplied field except customer_id; one duplicate from each pair is excluded pending reconciliation."),
        ("Exact full-row duplicates", int(raw.duplicated().sum()), "No full-row duplicates because the copied records carry different customer IDs."),
        ("Repeated customer IDs", int(raw["customer_id"].duplicated(keep=False).sum()), "No repeated IDs in this extract. Production splitting must still group by customer if repeats later appear."),
        ("Implausible employment tenure", int(invalid_employment.sum()), "55 values exceed age minus 14. Employment years is excluded until source correction."),
        ("Missing application income", int(raw["annual_income_at_application"].isna().sum()), f"{missing_rates['annual_income_at_application']:.1%}; retained with a training-only indicator and median imputation."),
        ("Missing credit score", int(raw["credit_score"].isna().sum()), f"{missing_rates['credit_score']:.1%}; imputation remains inside each training fold."),
        ("Missing utilization", int(raw["credit_utilization"].isna().sum()), f"{missing_rates['credit_utilization']:.1%}; missing among defaults {default_missing_util:.1%} versus {nondefault_missing_util:.1%} among non-defaults."),
        ("Income range", "$18,000 to $350,000", "Extreme incomes are plausible in context and are not mechanically deleted or winsorized."),
        ("Credit score range", "497 to 840", "Values remain inside the expected bureau-score range."),
        ("Geography cardinality", f"12 states; {raw['zip_prefix'].nunique()} ZIP prefixes", "ZIP target encoding would be unstable and creates proxy/redlining risk."),
        ("Unique identifier cardinality", raw["customer_id"].nunique(), "customer_id is unique and must not be converted into a predictor."),
        ("Observed default rate, 2022", f"{year_rates.loc[2022]:.2%}", "The event rate changes over calendar time."),
        ("Observed default rate, 2024", f"{year_rates.loc[2024]:.2%}", "Higher than 2022; random splitting would hide this shift."),
        ("2025 label availability", f"{raw.loc[raw.application_date.dt.year.eq(2025), 'label_available'].mean():.1%}", "All 2025 applications are censored in this extract."),
        ("Development partition", len(development), f"{int(development[TARGET].sum())} defaults; {development.application_date.min().date()} to {development.application_date.max().date()}."),
        ("Validation partition", len(validation), f"{int(validation[TARGET].sum())} defaults; {validation.application_date.min().date()} to {validation.application_date.max().date()}."),
        ("Final out-of-time test", len(final_test), f"{int(final_test[TARGET].sum())} defaults; {final_test.application_date.min().date()} to {final_test.application_date.max().date()}."),
        ("Bad-test train overlap", bad_result["test_contamination_count"], f"{bad_result['test_contamination_rate']:.1%} of bad-test rows reuse an original record seen in training; all {bad_result['contaminated_default_count']} bad-test defaults are contaminated."),
        ("Post-decision direct proxies", "days_past_due_6m_after; charged_off_amount_12m; account_status_12m", "These variables directly reveal the future outcome and are excluded."),
        ("Fairness review fields", "age; state; zip_prefix; home_ownership; channel", "Age and geography are excluded; retained application fields require ongoing proxy and subgroup monitoring."),
    ]
    return pd.DataFrame(rows, columns=["audit_item", "result", "interpretation"])


def subgroup_analysis(final_test, probability, threshold):
    frame = final_test[["customer_id", "age", TARGET]].copy()
    frame["probability_default"] = probability
    frame["prediction"] = (probability >= threshold).astype(int)
    frame["age_band"] = pd.cut(
        frame["age"], bins=[18, 35, 50, 101], right=False, labels=["19-34", "35-49", "50+"]
    )
    rows = []
    for age_band, group in frame.groupby("age_band", observed=False):
        y_true = group[TARGET].to_numpy(dtype=int)
        predicted = group["prediction"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        rows.append(
            {
                "age_band": str(age_band),
                "rows": len(group),
                "defaults": int(y_true.sum()),
                "default_rate": float(y_true.mean()) if len(group) else np.nan,
                "mean_predicted_probability": float(group["probability_default"].mean()),
                "roc_auc": safe_auc(y_true, group["probability_default"].to_numpy()),
                "pr_auc": float(average_precision_score(y_true, group["probability_default"]))
                if y_true.sum() > 0
                else np.nan,
                "brier_score": float(brier_score_loss(y_true, group["probability_default"])),
                "high_risk_flag_rate": float(predicted.mean()),
                "precision": float(precision_score(y_true, predicted, zero_division=0)),
                "recall": float(recall_score(y_true, predicted, zero_division=0)),
                "false_positive_rate": float(fp / (fp + tn)) if fp + tn else np.nan,
                "confusion_matrix": f"[[{tn}, {fp}], [{fn}, {tp}]]",
            }
        )
    return pd.DataFrame(rows)


def temporal_table(deduplicated):
    frame = deduplicated.copy()
    frame["quarter"] = frame["application_date"].dt.to_period("Q").astype(str)
    return (
        frame.groupby("quarter", as_index=False)
        .agg(
            rows=("customer_id", "size"),
            label_available_rate=("label_available", "mean"),
            observed_labels=(TARGET, "count"),
            defaults=(TARGET, "sum"),
            observed_default_rate=(TARGET, "mean"),
            median_months_observed=("months_observed", "median"),
        )
        .sort_values("quarter")
    )


def save_figures(output_dir, results, final_test, corrected_probability, temporal):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 150,
        }
    )
    colors = {"Submitted bad model": "#B3261E", "Corrected OOT model": "#1F4E79"}
    paths = []

    metric_columns = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    metric_labels = ["ROC AUC", "PR AUC", "Precision", "Recall", "F1"]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x_position = np.arange(len(metric_columns))
    width = 0.36
    for offset, (_, row) in zip([-width / 2, width / 2], results.iterrows()):
        values = [row[column] for column in metric_columns]
        bars = ax.bar(
            x_position + offset,
            values,
            width,
            label=row["model"],
            color=colors[row["model"]],
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xticks(x_position, metric_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Metric value")
    ax.set_title("Contaminated random test versus final out-of-time test")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "Liang_bad_vs_corrected_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path.name)

    y_test = final_test[TARGET].to_numpy(dtype=int)
    fpr, tpr, _ = roc_curve(y_test, corrected_probability)
    precision, recall, _ = precision_recall_curve(y_test, corrected_probability)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    axes[0].plot(fpr, tpr, color="#1F4E79", linewidth=2)
    axes[0].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].text(0.58, 0.12, f"AUC = {roc_auc_score(y_test, corrected_probability):.3f}")
    axes[1].plot(recall, precision, color="#1F4E79", linewidth=2)
    axes[1].axhline(y_test.mean(), color="#777777", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1].text(
        0.48,
        0.72,
        f"AP = {average_precision_score(y_test, corrected_probability):.3f}\nBaseline = {y_test.mean():.3f}",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    fig.suptitle("Corrected model on the untouched final test")
    fig.tight_layout()
    path = output_dir / "Liang_corrected_oot_roc_pr.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path.name)

    observed_rate, mean_probability = calibration_curve(
        y_test, corrected_probability, n_bins=5, strategy="quantile"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    axes[0].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    axes[0].plot(
        mean_probability,
        observed_rate,
        marker="o",
        color="#1F4E79",
        linewidth=2,
    )
    upper = max(0.15, float(max(observed_rate.max(), mean_probability.max()) * 1.15))
    axes[0].set_xlim(0, upper)
    axes[0].set_ylim(0, upper)
    axes[0].set(
        xlabel="Mean predicted default probability",
        ylabel="Observed default rate",
        title="Five quantile calibration bins",
    )
    axes[0].grid(alpha=0.25)
    axes[1].hist(corrected_probability, bins=14, color="#1F4E79", alpha=0.85)
    axes[1].set(
        xlabel="Predicted default probability",
        ylabel="Borrowers",
        title="Final-test score distribution",
    )
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"Calibration diagnostic: Brier score {brier_score_loss(y_test, corrected_probability):.3f}"
    )
    fig.tight_layout()
    path = output_dir / "Liang_corrected_calibration.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path.name)

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.7), sharex=True)
    quarter_position = np.arange(len(temporal))
    axes[0].bar(
        quarter_position,
        temporal["label_available_rate"] * 100,
        color="#6B8EAD",
    )
    axes[0].set_ylabel("Labels available (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Twelve-month label maturity collapses in recent quarters")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot(
        quarter_position,
        temporal["observed_default_rate"] * 100,
        marker="o",
        color="#B3261E",
        linewidth=2,
    )
    axes[1].set_ylabel("Observed default rate (%)")
    axes[1].set_xlabel("Application quarter")
    axes[1].grid(alpha=0.25)
    axes[1].set_xticks(quarter_position, temporal["quarter"], rotation=45, ha="right")
    for x_value, row in temporal.iterrows():
        if pd.notna(row["observed_default_rate"]):
            axes[1].annotate(
                f"n={int(row['observed_labels'])}",
                (x_value, row["observed_default_rate"] * 100),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
            )
    fig.tight_layout()
    path = output_dir / "Liang_label_maturity_and_temporal_default_rate.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path.name)
    return paths


def calibration_intercept_slope(y_true, probability):
    """Cox recalibration: regress the outcome on the score's logit.

    Slope near 1 means the spread of the score is right. A non-zero intercept
    at slope 1 means the whole level is shifted.
    """
    y_array = np.asarray(y_true, dtype=int)
    clipped = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    if np.unique(y_array).size < 2:
        return float("nan"), float("nan")
    fitted = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    fitted.fit(logit, y_array)
    return float(fitted.intercept_[0]), float(fitted.coef_[0][0])


def calibration_summary(validation, validation_probability, final_test, final_probability):
    rows = []
    for name, part, prob in [
        ("validation", validation, validation_probability),
        ("final out-of-time test", final_test, final_probability),
    ]:
        y_array = part[TARGET].to_numpy(dtype=int)
        intercept, slope = calibration_intercept_slope(y_array, prob)
        base_rate = float(y_array.mean())
        rows.append(
            {
                "sample": name,
                "rows": len(y_array),
                "defaults": int(y_array.sum()),
                "observed_default_rate": base_rate,
                "mean_predicted_probability": float(np.mean(prob)),
                "expected_over_observed": float(np.sum(prob) / y_array.sum())
                if y_array.sum()
                else np.nan,
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "brier_score": float(brier_score_loss(y_array, prob)),
                "no_skill_brier_at_base_rate": float(np.mean((y_array - base_rate) ** 2)),
            }
        )
    return pd.DataFrame(rows)


NEAR_DUPLICATE_FIELDS = [
    "age",
    "annual_income_at_application",
    "credit_score",
    "debt_to_income",
    "loan_amount",
    "interest_rate",
    "credit_utilization",
    "num_open_accounts",
]


def near_duplicate_screen(raw):
    """Count exact duplicates, then check for rows that are close but not identical."""
    signature = [column for column in raw.columns if column != "customer_id"]
    exact_mask = raw.duplicated(signature, keep=False)
    frame = raw[NEAR_DUPLICATE_FIELDS].fillna(raw[NEAR_DUPLICATE_FIELDS].median())
    scaled = StandardScaler().fit_transform(frame)
    distances = NearestNeighbors(n_neighbors=2).fit(scaled).kneighbors(scaled)[0][:, 1]

    survivors = ~raw.duplicated(signature, keep="first")
    frame_after = frame.loc[survivors]
    scaled_after = StandardScaler().fit_transform(frame_after)
    distances_after = NearestNeighbors(n_neighbors=2).fit(scaled_after).kneighbors(scaled_after)[0][:, 1]

    rows = [
        {
            "check": "Exact duplicate pairs (all fields except customer_id)",
            "value": int(exact_mask.sum() // 2),
            "note": "One record of each pair is excluded from the modeling population.",
        },
        {
            "check": "Median nearest-neighbor distance (standardized, 8 continuous fields)",
            "value": round(float(np.median(distances)), 4),
            "note": "Typical separation between two different borrowers.",
        },
    ]
    for cutoff in [0.01, 0.05, 0.10, 0.25]:
        rows.append(
            {
                "check": f"Rows with a neighbor within {cutoff:.2f}",
                "value": int((distances < cutoff).sum()),
                "note": "Counted before the exact duplicates are removed.",
            }
        )
    rows.append(
        {
            "check": "Rows with a neighbor within 0.10 after removing exact duplicates",
            "value": int((distances_after < 0.10).sum()),
            "note": "Zero means the duplication is copying, not a cluster of similar borrowers.",
        }
    )
    return pd.DataFrame(rows)


def outlier_screen(raw):
    """Per-variable range and rule screen. Nothing is dropped based on this table alone."""
    numeric_fields = [
        "age",
        "annual_income_at_application",
        "employment_years",
        "credit_score",
        "debt_to_income",
        "loan_amount",
        "interest_rate",
        "credit_utilization",
        "num_open_accounts",
        "delinq_2y",
        "inq_6m",
    ]
    rows = []
    for field in numeric_fields:
        values = raw[field].dropna()
        rows.append(
            {
                "field": field,
                "missing": int(raw[field].isna().sum()),
                "minimum": float(values.min()),
                "p1": float(values.quantile(0.01)),
                "p99": float(values.quantile(0.99)),
                "maximum": float(values.max()),
                "rule_violations": int((raw["employment_years"] > raw["age"] - 14).sum())
                if field == "employment_years"
                else 0,
                "treatment": "Excluded until the source is corrected."
                if field == "employment_years"
                else "Retained; extreme but plausible for a lending population.",
            }
        )
    return pd.DataFrame(rows)


def extended_subgroup_analysis(final_test, probability, threshold):
    """Subgroup results for home ownership, channel and state. Age bands are
    reported separately in subgroup_analysis."""
    frame = final_test[["customer_id", "home_ownership", "channel", "state", TARGET]].copy()
    frame["probability_default"] = probability
    frame["prediction"] = (probability >= threshold).astype(int)
    rows = []
    for dimension in ["home_ownership", "channel", "state"]:
        for level, group in frame.groupby(dimension, observed=False):
            y_true = group[TARGET].to_numpy(dtype=int)
            predicted = group["prediction"].to_numpy(dtype=int)
            rows.append(
                {
                    "dimension": dimension,
                    "level": str(level),
                    "rows": len(group),
                    "defaults": int(y_true.sum()),
                    "default_rate": float(y_true.mean()),
                    "mean_predicted_probability": float(group["probability_default"].mean()),
                    "high_risk_flag_rate": float(predicted.mean()),
                    "roc_auc": safe_auc(y_true, group["probability_default"].to_numpy()),
                    "readable": "no, cell too small" if len(group) < 30 or y_true.sum() < 3 else "yes",
                }
            )
    return pd.DataFrame(rows)


def challenger_benchmarks(development, final_test, selected, champion_probability):
    """Compare the champion with a credit_score-only benchmark, and measure what
    excluding employment_years and interest_rate cost."""
    y_final = final_test[TARGET].to_numpy(dtype=int)
    rows = [
        {
            "model": "Champion: 16 application-time features",
            "features": len(MODEL_FEATURES),
            "roc_auc": safe_auc(y_final, champion_probability),
            "pr_auc": float(average_precision_score(y_final, champion_probability)),
            "brier_score": float(brier_score_loss(y_final, champion_probability)),
        }
    ]

    single = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=selected["C"],
                    class_weight=selected["class_weight"],
                    max_iter=5000,
                    solver="liblinear",
                    random_state=CORRECTED_SEED,
                ),
            ),
        ]
    )
    single.fit(development[["credit_score"]], development[TARGET])
    single_probability = single.predict_proba(final_test[["credit_score"]])[:, 1]
    rows.append(
        {
            "model": "Challenger: credit_score alone",
            "features": 1,
            "roc_auc": safe_auc(y_final, single_probability),
            "pr_auc": float(average_precision_score(y_final, single_probability)),
            "brier_score": float(brier_score_loss(y_final, single_probability)),
        }
    )

    wider_numeric = MODEL_NUMERIC + ["employment_years", "interest_rate"]
    wider_features = wider_numeric + MODEL_CATEGORICAL
    wider = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            wider_numeric,
                        ),
                        (
                            "categorical",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            MODEL_CATEGORICAL,
                        ),
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=selected["C"],
                    class_weight=selected["class_weight"],
                    max_iter=5000,
                    solver="liblinear",
                    random_state=CORRECTED_SEED,
                ),
            ),
        ]
    )
    wider.fit(development[wider_features], development[TARGET])
    wider_probability = wider.predict_proba(final_test[wider_features])[:, 1]
    rows.append(
        {
            "model": "Sensitivity: champion plus employment_years and interest_rate",
            "features": len(wider_features),
            "roc_auc": safe_auc(y_final, wider_probability),
            "pr_auc": float(average_precision_score(y_final, wider_probability)),
            "brier_score": float(brier_score_loss(y_final, wider_probability)),
        }
    )
    return pd.DataFrame(rows)


LEAKED_FIELDS = [
    "days_past_due_6m_after",
    "collections_contacted_9m_after",
    "charged_off_amount_12m",
    "account_status_12m",
    "current_balance",
    "current_income",
    "months_observed",
    "label_available",
]


def _attribution_frame(raw, drop_leaked, drop_censored):
    """The submitted feature build, with the population switches exposed."""
    df = raw.copy()
    if drop_censored:
        df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].fillna(0).astype(int)
    df["customer_number"] = df["customer_id"].str.replace("CUST-", "", regex=False).astype(int)
    df["application_year"] = df["application_date"].dt.year
    df["application_month"] = df["application_date"].dt.month
    df["application_day"] = df["application_date"].dt.day
    df["application_dayofweek"] = df["application_date"].dt.dayofweek
    df["loan_to_income"] = df["loan_amount"] / df["annual_income_at_application"].replace(0, np.nan)
    df["balance_to_loan"] = df["current_balance"] / df["loan_amount"].replace(0, np.nan)
    df["income_change"] = (
        (df["current_income"] - df["annual_income_at_application"])
        / df["annual_income_at_application"].replace(0, np.nan)
    )
    df = df.sort_values("application_date", kind="stable").reset_index(drop=True)
    df = df.drop(columns=["application_date", "customer_id"])
    if drop_leaked:
        remove = [c for c in LEAKED_FIELDS + ["balance_to_loan", "income_change"] if c in df]
        df = df.drop(columns=remove)
    return df


def _prep_whole_sample(df):
    """Preprocessing fitted on everything, exactly as the submission does it."""
    df = df.copy()
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    df[numeric] = SimpleImputer(strategy="median").fit_transform(df[numeric])
    if categorical:
        df[categorical] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical])
    for column in categorical:
        df[column + "_target_rate"] = df[column].map(df.groupby(column)[TARGET].mean())
    df = pd.get_dummies(df, columns=categorical, drop_first=True)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median(numeric_only=True)).fillna(0)
    target = df[TARGET]
    features = df.drop(columns=[TARGET])
    scaled = StandardScaler().fit_transform(features)
    return pd.DataFrame(scaled, columns=features.columns), target


def _prep_training_only(train, test):
    """Same steps, but every statistic comes from the training rows."""
    train, test = train.copy(), test.copy()
    numeric = [c for c in train.select_dtypes(include=[np.number]).columns if c != TARGET]
    categorical = train.select_dtypes(exclude=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy="median").fit(train[numeric])
    train[numeric], test[numeric] = imputer.transform(train[numeric]), imputer.transform(test[numeric])
    if categorical:
        cat_imputer = SimpleImputer(strategy="most_frequent").fit(train[categorical])
        train[categorical] = cat_imputer.transform(train[categorical])
        test[categorical] = cat_imputer.transform(test[categorical])
    for column in categorical:
        rate = train.groupby(column)[TARGET].mean()
        train[column + "_target_rate"] = train[column].map(rate)
        test[column + "_target_rate"] = test[column].map(rate).fillna(train[TARGET].mean())
    combined = pd.get_dummies(
        pd.concat([train.assign(_split=0), test.assign(_split=1)]), columns=categorical, drop_first=True
    ).replace([np.inf, -np.inf], np.nan)
    train_out = combined[combined._split == 0].drop(columns="_split")
    test_out = combined[combined._split == 1].drop(columns="_split")
    medians = train_out.median(numeric_only=True)
    train_out, test_out = train_out.fillna(medians).fillna(0), test_out.fillna(medians).fillna(0)
    y_train, y_test = train_out[TARGET], test_out[TARGET]
    x_train, x_test = train_out.drop(columns=[TARGET]), test_out.drop(columns=[TARGET])
    scaler = StandardScaler().fit(x_train)
    return (
        pd.DataFrame(scaler.transform(x_train), columns=x_train.columns),
        y_train.reset_index(drop=True),
        pd.DataFrame(scaler.transform(x_test), columns=x_test.columns),
        y_test.reset_index(drop=True),
    )


def _oversample(features, target, rng):
    features, target = features.reset_index(drop=True), target.reset_index(drop=True)
    defaults = target[target == 1].index.to_numpy()
    others = target[target == 0].index.to_numpy()
    if len(defaults) >= len(others):
        return features, target
    extra = rng.choice(defaults, size=len(others) - len(defaults), replace=True)
    index = np.concatenate([others, defaults, extra])
    rng.shuffle(index)
    return features.loc[index].reset_index(drop=True), target.loc[index].reset_index(drop=True)


def _screen(features, target, keep=35):
    scores = mutual_info_classif(features, target, random_state=BAD_SEED)
    return features.columns[np.argsort(scores)[::-1][:min(keep, features.shape[1])]].tolist()


def _auc_of_run(x_train, y_train, x_test, y_test):
    model = GradientBoostingClassifier(
        n_estimators=250, learning_rate=0.04, max_depth=3, random_state=BAD_SEED
    )
    model.fit(x_train, y_train)
    return safe_auc(y_test, model.predict_proba(x_test)[:, 1]), len(y_test), int(y_test.sum())


def leakage_attribution(raw, seeds):
    """Repair one fault at a time and record the resulting test score.

    The model family is held fixed so any movement is attributable to the
    repair rather than to a different estimator.
    """
    rows = []

    # Part 1: are leakage and contamination each sufficient on their own?
    for keep_leaked in [True, False]:
        for oversample_first in [True, False]:
            scores = []
            for seed in seeds:
                df = _attribution_frame(raw, drop_leaked=not keep_leaked, drop_censored=False)
                features, target = _prep_whole_sample(df)
                rng = np.random.RandomState(seed)
                if oversample_first:
                    balanced, balanced_target = _oversample(features, target, rng)
                    balanced = balanced[_screen(balanced, balanced_target)]
                    x_train, x_test, y_train, y_test = train_test_split(
                        balanced, balanced_target, test_size=0.25, random_state=seed, shuffle=True
                    )
                else:
                    x_train, x_test, y_train, y_test = train_test_split(
                        features, target, test_size=0.25, random_state=seed, shuffle=True
                    )
                    x_train, y_train = _oversample(x_train, y_train, rng)
                    keep = _screen(x_train, y_train)
                    x_train, x_test, y_test = x_train[keep], x_test[keep], y_test.reset_index(drop=True)
                scores.append(_auc_of_run(x_train, y_train, x_test, y_test)[0])
            rows.append(
                {
                    "study": "sufficiency 2x2",
                    "step": f"leaked fields {'kept' if keep_leaked else 'removed'}; "
                            f"oversampling {'before' if oversample_first else 'after'} the split",
                    "test_roc_auc": round(float(np.mean(scores)), 4),
                    "seed_minimum": round(float(np.min(scores)), 4),
                    "seed_maximum": round(float(np.max(scores)), 4),
                    "test_rows": np.nan,
                    "test_defaults": np.nan,
                }
            )

    # Part 2: cumulative repair ladder, one seed, reported in order.
    seed = seeds[0]
    ladder = []

    df = _attribution_frame(raw, drop_leaked=False, drop_censored=False)
    features, target = _prep_whole_sample(df)
    balanced, balanced_target = _oversample(features, target, np.random.RandomState(seed))
    balanced = balanced[_screen(balanced, balanced_target)]
    split = train_test_split(balanced, balanced_target, test_size=0.25, random_state=seed, shuffle=True)
    ladder.append(("R0 as submitted", _auc_of_run(split[0], split[2], split[1], split[3])))

    df = _attribution_frame(raw, drop_leaked=True, drop_censored=False)
    features, target = _prep_whole_sample(df)
    balanced, balanced_target = _oversample(features, target, np.random.RandomState(seed))
    balanced = balanced[_screen(balanced, balanced_target)]
    split = train_test_split(balanced, balanced_target, test_size=0.25, random_state=seed, shuffle=True)
    ladder.append(("R1 drop leaked and post-decision fields", _auc_of_run(split[0], split[2], split[1], split[3])))

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=seed, shuffle=True
    )
    x_train, y_train = _oversample(x_train, y_train, np.random.RandomState(seed))
    keep = _screen(x_train, y_train)
    ladder.append(
        ("R2 oversample after the split", _auc_of_run(x_train[keep], y_train, x_test[keep], y_test.reset_index(drop=True)))
    )

    for label, drop_censored, chronological in [
        ("R3 fit preprocessing and screening on training rows only", False, False),
        ("R4 drop the censored rows instead of relabeling them", True, False),
        ("R5 split chronologically instead of at random", True, True),
    ]:
        df = _attribution_frame(raw, drop_leaked=True, drop_censored=drop_censored)
        if chronological:
            cut = int(len(df) * 0.75)
            train, test = df.iloc[:cut], df.iloc[cut:]
        else:
            train, test = train_test_split(df, test_size=0.25, random_state=seed, shuffle=True)
        x_train, y_train, x_test, y_test = _prep_training_only(train, test)
        x_train, y_train = _oversample(x_train, y_train, np.random.RandomState(seed))
        keep = _screen(x_train, y_train)
        ladder.append((label, _auc_of_run(x_train[keep], y_train, x_test[keep], y_test)))

    for label, (auc, rows_count, defaults) in ladder:
        rows.append(
            {
                "study": "cumulative repair ladder",
                "step": label,
                "test_roc_auc": round(float(auc), 4),
                "seed_minimum": np.nan,
                "seed_maximum": np.nan,
                "test_rows": rows_count,
                "test_defaults": defaults,
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    data_path = resolve_data_path(args.data)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_path)
    raw["application_date"] = pd.to_datetime(raw["application_date"], errors="raise")
    expected_columns = set(FIELD_CLASSIFICATION)
    if set(raw.columns) != expected_columns:
        missing = expected_columns - set(raw.columns)
        extra = set(raw.columns) - expected_columns
        raise ValueError(f"Unexpected schema. Missing={sorted(missing)} extra={sorted(extra)}")

    bad_result = reproduce_bad_model(raw)
    deduplicated, eligible, audit_counts = prepare_population(raw)
    development, validation, final_test, dev_boundary, test_boundary = chronological_split(
        eligible
    )
    corrected = corrected_model_results(
        development, validation, final_test, args.bootstrap_reps
    )

    bad_metrics = bad_result["metrics"]
    corrected_metrics = corrected["metrics"]
    corrected_intervals = corrected["intervals"]
    results = pd.DataFrame(
        [
            {
                "model": "Submitted bad model",
                "modeling_population_size": bad_result["modeling_population_size"],
                "evaluation_sample_size": bad_result["evaluation_size"],
                "evaluation_defaults": bad_result["evaluation_defaults"],
                "default_prevalence": bad_result["evaluation_prevalence"],
                "raw_feature_count": bad_result["feature_count"],
                "feature_count": bad_result["feature_count"],
                "split_method": "Random row split after full-data preprocessing, target encoding, feature selection, and oversampling",
                "threshold_policy": "Maximize accuracy on the same test set",
                "threshold": bad_result["threshold"],
                "roc_auc": bad_metrics["roc_auc"],
                "roc_auc_reported_binary": bad_metrics["roc_auc_reported_binary"],
                "pr_auc": bad_metrics["pr_auc"],
                "accuracy": bad_metrics["accuracy"],
                "precision": bad_metrics["precision"],
                "recall": bad_metrics["recall"],
                "f1": bad_metrics["f1"],
                "confusion_matrix": bad_metrics["confusion_matrix"],
                "true_negative": bad_metrics["true_negative"],
                "false_positive": bad_metrics["false_positive"],
                "false_negative": bad_metrics["false_negative"],
                "true_positive": bad_metrics["true_positive"],
                "brier_score": bad_metrics["brier_score"],
                "roc_auc_ci_low": np.nan,
                "roc_auc_ci_high": np.nan,
                "pr_auc_ci_low": np.nan,
                "pr_auc_ci_high": np.nan,
                "brier_ci_low": np.nan,
                "brier_ci_high": np.nan,
                "test_rows_also_in_training": bad_result["test_contamination_count"],
                "notes": "Not credible: future outcome fields and repeated oversampled records contaminate the test set.",
            },
            {
                "model": "Corrected OOT model",
                "modeling_population_size": len(eligible),
                "evaluation_sample_size": len(final_test),
                "evaluation_defaults": int(final_test[TARGET].sum()),
                "default_prevalence": float(final_test[TARGET].mean()),
                "raw_feature_count": len(MODEL_FEATURES),
                "feature_count": corrected["transformed_feature_count"],
                "split_method": f"Chronological: develop before {dev_boundary.date()}, validate before {test_boundary.date()}, then final OOT test",
                "threshold_policy": "Highest validation threshold retaining 100% recall; final test untouched",
                "threshold": corrected["threshold"],
                "roc_auc": corrected_metrics["roc_auc"],
                "roc_auc_reported_binary": np.nan,
                "pr_auc": corrected_metrics["pr_auc"],
                "accuracy": corrected_metrics["accuracy"],
                "precision": corrected_metrics["precision"],
                "recall": corrected_metrics["recall"],
                "f1": corrected_metrics["f1"],
                "confusion_matrix": corrected_metrics["confusion_matrix"],
                "true_negative": corrected_metrics["true_negative"],
                "false_positive": corrected_metrics["false_positive"],
                "false_negative": corrected_metrics["false_negative"],
                "true_positive": corrected_metrics["true_positive"],
                "brier_score": corrected_metrics["brier_score"],
                **corrected_intervals,
                "test_rows_also_in_training": 0,
                "notes": "Leakage-free benchmark. Nine final-test defaults make all estimates imprecise; threshold requires business-cost validation.",
            },
        ]
    )

    audit = build_data_audit(
        raw,
        deduplicated,
        eligible,
        audit_counts,
        development,
        validation,
        final_test,
        bad_result,
    )
    dictionary = data_dictionary(raw)
    subgroup = subgroup_analysis(
        final_test, corrected["final_probability"], corrected["threshold"]
    )
    temporal = temporal_table(deduplicated)
    calibration = calibration_summary(
        validation, corrected["validation_probability"], final_test, corrected["final_probability"]
    )
    near_duplicates = near_duplicate_screen(raw)
    outliers = outlier_screen(raw)
    extended_subgroup = extended_subgroup_analysis(
        final_test, corrected["final_probability"], corrected["threshold"]
    )
    challengers = challenger_benchmarks(
        development, final_test, corrected["selected"], corrected["final_probability"]
    )
    attribution = leakage_attribution(
        raw, seeds=[BAD_SEED + offset for offset in range(args.attribution_seeds)]
    )
    predictions = final_test[
        ["customer_id", "application_date", "age", "state", TARGET]
    ].copy()
    predictions["probability_default"] = corrected["final_probability"]
    predictions["threshold"] = corrected["threshold"]
    predictions["predicted_default"] = (
        predictions["probability_default"] >= corrected["threshold"]
    ).astype(int)

    results.to_csv(output_dir / "Liang_results.csv", index=False)
    audit.to_csv(output_dir / "Liang_data_audit.csv", index=False)
    dictionary.to_csv(
        output_dir / "Liang_data_dictionary.csv", index=False
    )
    corrected["tuning"].to_csv(
        output_dir / "Liang_cv_results.csv", index=False
    )
    subgroup.to_csv(
        output_dir / "Liang_subgroup_analysis.csv", index=False
    )
    temporal.to_csv(
        output_dir / "Liang_temporal_analysis.csv", index=False
    )
    predictions.to_csv(
        output_dir / "Liang_final_test_predictions.csv", index=False
    )
    bad_result["top_mi"].to_csv(
        output_dir / "Liang_bad_model_top_features.csv", index=False
    )
    calibration.to_csv(output_dir / "Liang_calibration_summary.csv", index=False)
    near_duplicates.to_csv(output_dir / "Liang_near_duplicate_screen.csv", index=False)
    outliers.to_csv(output_dir / "Liang_outlier_screen.csv", index=False)
    extended_subgroup.to_csv(output_dir / "Liang_subgroup_extended.csv", index=False)
    challengers.to_csv(output_dir / "Liang_challenger_benchmarks.csv", index=False)
    attribution.to_csv(output_dir / "Liang_leakage_attribution.csv", index=False)

    figure_files = save_figures(
        output_dir,
        results,
        final_test,
        corrected["final_probability"],
        temporal,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": data_path.name,
        "input_sha256": sha256_file(data_path),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "bad_seed": BAD_SEED,
        "corrected_seed": CORRECTED_SEED,
        "eligible_definition": "label_available=1, months_observed>=12, nonmissing default_12m, probable duplicate removed",
        "model_features": MODEL_FEATURES,
        "selected_hyperparameters": corrected["selected"],
        "threshold": corrected["threshold"],
        "development_boundary": str(dev_boundary.date()),
        "final_test_boundary": str(test_boundary.date()),
        "figure_files": figure_files,
        "final_test_calibration_slope": float(
            calibration.loc[1, "calibration_slope"]
        ),
        "final_test_expected_over_observed": float(
            calibration.loc[1, "expected_over_observed"]
        ),
    }
    (output_dir / "Liang_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Completed")
    print(results.to_string(index=False))
    print(f"\nSelected corrected parameters: {corrected['selected']}")
    print(f"Validation-only threshold: {corrected['threshold']:.6f}")
    print(f"Input SHA-256: {manifest['input_sha256']}")
    print(
        "Final-test calibration: E/O %.3f, slope %.3f"
        % (calibration.loc[1, "expected_over_observed"], calibration.loc[1, "calibration_slope"])
    )


if __name__ == "__main__":
    main()
