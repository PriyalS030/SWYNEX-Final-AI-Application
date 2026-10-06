"""
train_model.py
==============
Train and evaluate classification models for Student Performance Risk Prediction.

Uses the UCI Student Performance Dataset (Mathematics).
Target: G3 < 10 -> At Risk (1), G3 >= 10 -> Low Risk (0)

Models trained:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. Support Vector Machine

Two experiments are conducted:
  - Model A: With G1 and G2 features
  - Model B: Without G1 and G2 features

The best model (based on At Risk recall and F1-score) is saved for the Flask app.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# ---------------------------------------------
# Configuration
# ---------------------------------------------
DATA_PATH = os.path.join('data', 'student-mat.csv')
MODEL_DIR = 'model'
RESULTS_DIR = 'results'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Features to use (from the user's specification)
FEATURES_WITH_G = [
    'studytime', 'failures', 'absences',
    'schoolsup', 'famsup', 'paid', 'activities',
    'higher', 'internet',
    'freetime', 'goout', 'health',
    'G1', 'G2'
]

FEATURES_WITHOUT_G = [
    'studytime', 'failures', 'absences',
    'schoolsup', 'famsup', 'paid', 'activities',
    'higher', 'internet',
    'freetime', 'goout', 'health'
]

# Binary yes/no columns that need encoding
BINARY_COLUMNS = ['schoolsup', 'famsup', 'paid', 'activities', 'higher', 'internet']


def load_and_inspect_data():
    """Load the UCI Student Performance dataset and display basic info."""
    print("=" * 60)
    print("PHASE 1: Loading and Inspecting Dataset")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH, sep=';')

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  {col}: dtype={df[col].dtype}, unique={df[col].nunique()}")

    print(f"\nFirst 5 rows:")
    print(df.head().to_string())

    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values found.")
    else:
        print(missing[missing > 0])

    print(f"\nG3 (Final Grade) statistics:")
    print(df['G3'].describe())

    return df


def prepare_data(df):
    """Create target variable and prepare features."""
    print("\n" + "=" * 60)
    print("PHASE 2: Preparing Data")
    print("=" * 60)

    # Create binary target: G3 < 10 -> At Risk (1), G3 >= 10 -> Low Risk (0)
    df['risk'] = (df['G3'] < 10).astype(int)

    print(f"\nTarget distribution (G3 < 10 -> At Risk):")
    print(f"  At Risk (1): {df['risk'].sum()} ({df['risk'].mean()*100:.1f}%)")
    print(f"  Low Risk (0): {(1 - df['risk']).sum()} ({(1 - df['risk']).mean()*100:.1f}%)")

    # Encode binary yes/no columns
    for col in BINARY_COLUMNS:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map({'yes': 1, 'no': 0})
            print(f"  Encoded '{col}': yes->1, no->0")

    # Verify all features are present
    print(f"\nFeatures with G1/G2 ({len(FEATURES_WITH_G)}):")
    for f in FEATURES_WITH_G:
        if f in df.columns:
            print(f"  OK {f}: {df[f].dtype}")
        else:
            print(f"  X {f}: NOT FOUND")

    return df


def get_models():
    """Return dictionary of models to train."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        'SVM': SVC(
            kernel='rbf', random_state=RANDOM_STATE
        )
    }


def train_and_evaluate(df, feature_list, experiment_name):
    """
    Train all models on the given features and return evaluation results.

    Returns:
        results: list of dicts with metrics for each model
        best_model_name: name of the best model
        best_model: trained best model object
        scaler: fitted StandardScaler
        X_test, y_test: test data for confusion matrix
    """
    print(f"\n{'=' * 60}")
    print(f"PHASE 3: Training & Evaluating Models — {experiment_name}")
    print(f"{'=' * 60}")

    X = df[feature_list].values
    y = df['risk'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    print(f"Train At Risk: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"Test At Risk:  {y_test.sum()} ({y_test.mean()*100:.1f}%)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    results = []
    best_f1_at_risk = -1
    best_model_name = None
    best_model = None

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        # Metrics for At Risk class (label=1)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Cross-validation score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
        cv_f1_mean = cv_scores.mean()

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision (At Risk)': prec,
            'Recall (At Risk)': rec,
            'F1 (At Risk)': f1,
            'CV F1 Mean': cv_f1_mean,
            'Confusion Matrix': cm
        })

        print(f"    Accuracy:           {acc:.4f}")
        print(f"    Precision (At Risk): {prec:.4f}")
        print(f"    Recall (At Risk):    {rec:.4f}")
        print(f"    F1 (At Risk):        {f1:.4f}")
        print(f"    CV F1 Mean:          {cv_f1_mean:.4f}")
        print(f"    Confusion Matrix:")
        print(f"      {cm}")

        print(f"\n    Full Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low Risk', 'At Risk']))

        # Select best model based on At Risk F1-score (with recall as tiebreaker)
        if f1 > best_f1_at_risk or (f1 == best_f1_at_risk and rec > best_recall_at_risk):
            best_f1_at_risk = f1
            best_recall_at_risk = rec
            best_model_name = name
            best_model = model

    print(f"\n  ► Best model for {experiment_name}: {best_model_name}")
    print(f"    F1 (At Risk) = {best_f1_at_risk:.4f}")

    return results, best_model_name, best_model, scaler, X_test, y_test


def save_results(results_a, results_b, best_a, best_b):
    """Save model comparison results to CSV and generate confusion matrix plots."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -- Save CSV comparison --
    rows = []
    for r in results_a:
        row = {k: v for k, v in r.items() if k != 'Confusion Matrix'}
        row['Experiment'] = 'With G1 & G2'
        rows.append(row)
    for r in results_b:
        row = {k: v for k, v in r.items() if k != 'Confusion Matrix'}
        row['Experiment'] = 'Without G1 & G2'
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n  Saved comparison to {csv_path}")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    # -- Generate confusion matrix plots --
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Confusion Matrices — All Models', fontsize=16, fontweight='bold')

    for idx, (results, title) in enumerate([(results_a, 'With G1 & G2'), (results_b, 'Without G1 & G2')]):
        for j, r in enumerate(results):
            ax = axes[idx][j]
            sns.heatmap(
                r['Confusion Matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low Risk', 'At Risk'],
                yticklabels=['Low Risk', 'At Risk'],
                ax=ax
            )
            ax.set_title(f"{r['Model']}\n({title})", fontsize=10)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrices to {cm_path}")

    # -- Generate bar chart comparison --
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (results, title) in enumerate([(results_a, 'With G1 & G2'), (results_b, 'Without G1 & G2')]):
        ax = axes[idx]
        model_names = [r['Model'] for r in results]
        metrics = ['Accuracy', 'Precision (At Risk)', 'Recall (At Risk)', 'F1 (At Risk)']
        x = np.arange(len(model_names))
        width = 0.2

        for k, metric in enumerate(metrics):
            values = [r[metric] for r in results]
            ax.bar(x + k * width, values, width, label=metric)

        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title(f'Model Comparison — {title}')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, 'model_comparison_chart.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison chart to {bar_path}")


def save_model(model, scaler, feature_list, model_name):
    """Save the selected model, scaler, feature list, and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_data = {
        'model': model,
        'scaler': scaler,
        'features': feature_list,
        'model_name': model_name,
        'binary_columns': BINARY_COLUMNS
    }

    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    joblib.dump(model_data, model_path)
    print(f"\n  Saved model to {model_path}")
    print(f"  Model: {model_name}")
    print(f"  Features: {feature_list}")


def run_example_predictions(model, scaler, feature_list):
    """Run example predictions to verify the model works."""
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)

    # Example 1: A student likely at risk
    example_1 = {
        'studytime': 1, 'failures': 2, 'absences': 15,
        'schoolsup': 0, 'famsup': 0, 'paid': 0,
        'activities': 0, 'higher': 0, 'internet': 1,
        'freetime': 4, 'goout': 5, 'health': 3,
        'G1': 5, 'G2': 5
    }

    # Example 2: A student likely low risk
    example_2 = {
        'studytime': 3, 'failures': 0, 'absences': 2,
        'schoolsup': 0, 'famsup': 1, 'paid': 1,
        'activities': 1, 'higher': 1, 'internet': 1,
        'freetime': 3, 'goout': 2, 'health': 4,
        'G1': 14, 'G2': 14
    }

    for i, example in enumerate([example_1, example_2], 1):
        # Filter to only the features used
        input_values = [example[f] for f in feature_list]
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        label = "At Risk" if prediction == 1 else "Low Risk"

        print(f"\n  Example {i}:")
        for f in feature_list:
            print(f"    {f}: {example[f]}")
        print(f"    -> Prediction: {label}")


def main():
    """Main training pipeline."""
    print("Student Performance Risk Prediction — Model Training")
    print("=" * 60)

    # Phase 1: Load and inspect
    df = load_and_inspect_data()

    # Phase 2: Prepare data
    df = prepare_data(df)

    # Phase 3 & 4: Train, evaluate, and select
    # Experiment A: With G1 and G2
    results_a, best_name_a, best_model_a, scaler_a, _, _ = train_and_evaluate(
        df, FEATURES_WITH_G, "Model A (With G1 & G2)"
    )

    # Experiment B: Without G1 and G2
    results_b, best_name_b, best_model_b, scaler_b, _, _ = train_and_evaluate(
        df, FEATURES_WITHOUT_G, "Model B (Without G1 & G2)"
    )

    # Save results
    save_results(results_a, results_b, best_name_a, best_name_b)

    # -- Select the overall best model --
    # Compare best from A and best from B
    best_a_f1 = max(r['F1 (At Risk)'] for r in results_a)
    best_b_f1 = max(r['F1 (At Risk)'] for r in results_b)

    print("\n" + "=" * 60)
    print("PHASE 4: Model Selection")
    print("=" * 60)
    print(f"\n  Best from Model A (With G1 & G2): {best_name_a} — F1={best_a_f1:.4f}")
    print(f"  Best from Model B (Without G1 & G2): {best_name_b} — F1={best_b_f1:.4f}")

    # Use Model A (with G1/G2) for the prototype since it's more accurate
    # But document the comparison in results
    if best_a_f1 >= best_b_f1:
        selected_model = best_model_a
        selected_scaler = scaler_a
        selected_features = FEATURES_WITH_G
        selected_name = best_name_a
        selected_experiment = "Model A (With G1 & G2)"
    else:
        selected_model = best_model_b
        selected_scaler = scaler_b
        selected_features = FEATURES_WITHOUT_G
        selected_name = best_name_b
        selected_experiment = "Model B (Without G1 & G2)"

    print(f"\n  ► Selected for prototype: {selected_name} from {selected_experiment}")

    # Save the selected model
    save_model(selected_model, selected_scaler, selected_features, selected_name)

    # Run example predictions
    run_example_predictions(selected_model, selected_scaler, selected_features)

    # Cleanup temp data
    import shutil
    if os.path.exists('data_temp'):
        shutil.rmtree('data_temp')
        print("\n  Cleaned up temporary data files.")

    print("\n" + "=" * 60)
    print("Training complete. Run 'python app.py' to start the web app.")
    print("=" * 60)


if __name__ == '__main__':
    main()
