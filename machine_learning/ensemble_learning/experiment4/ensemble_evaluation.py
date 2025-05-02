import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
# Potentially import models/functions from previous experiments if needed

# --- Configuration ---
RANDOM_STATE = 42
N_SPLITS_STACKING = 5  # Number of folds for generating OOF predictions if needed

# --- Paths ---
# Data paths
FEATURE_ENGINEERING_PATH = "../feature_engineering/feature_engineering_features.pkl"
FEATURE_NAMES_PATH = "../models/feature_engineering_keys.pkl"
HASH_LIST_PATH = "../models/hash_list.pkl"
BLACK_LIST_PATH = "../models/black_list.pkl"

# Base model paths (NEEDS CONFIRMATION FROM USER)
BASE_MODEL_PATHS = {
    'model1': '../experiment/results/lgbm_models.pkl', # Example path, confirm!
    'model2': '../experiment2/results/model.pkl',      # Example path, confirm!
    'model3': '../experiment3/results/lgbm_models.pkl', # Example path, confirm!
}

# Base model metrics paths (NEEDS CONFIRMATION FROM USER)
BASE_METRICS_PATHS = {
    'model1': '../experiment/results/model_metrics.csv', # Example path, confirm!
    'model2': '../experiment2/results/metrics.csv',      # Example path, confirm!
    'model3': '../experiment3/results/model_metrics.csv', # Example path, confirm!
}

# Experiment 4 output paths
SAVE_DIR = './experiment4/results'
FIGURES_DIR = './experiment4/figures'

# --- Helper Functions ---

def load_data():
    """Loads features, labels, and feature names."""
    print("Loading data...")
    with open(FEATURE_ENGINEERING_PATH, 'rb') as f:
        features = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    with open(HASH_LIST_PATH, "rb") as f:
        hash_list = pickle.load(f)
    with open(BLACK_LIST_PATH, "rb") as f:
        black_list = pickle.load(f)

    labels = np.array([1 if h in black_list else 0 for h in hash_list], dtype=np.int32)
    features = np.array(features, dtype=np.float32)

    # Clean data (important before feeding to models)
    print("Cleaning data (handling NaN/inf)...")
    features = np.nan_to_num(features, nan=0.0,
                             posinf=np.finfo(np.float32).max,
                             neginf=np.finfo(np.float32).min)

    print(f"Data loaded: Features shape={features.shape}, Labels count={len(labels)}")
    print(f"Label distribution: Benign={np.sum(labels==0)}, Malicious={np.sum(labels==1)}")
    return features, labels, feature_names

def load_model(path):
    """Loads a pickled model."""
    print(f"Loading model from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        # Handle potential list of k-fold models
        if isinstance(model, list):
            print(f"Loaded {len(model)} models (likely k-fold). Will use the first one for simplicity or average predictions.")
            # Decide strategy: use first model, average predictions, etc.
            # For simplicity, let's use the first model for now if it's a list
            # return model[0]
            # Or return the list to handle averaging in prediction function
            return model
        return model
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        raise

def get_predictions(model, X):
    """Generates predictions handling different model types/outputs."""
    if isinstance(model, list): # Handle k-fold models by averaging predictions
        print(f"Averaging predictions from {len(model)} k-fold models...")
        all_preds = []
        for sub_model in model:
            try:
                if hasattr(sub_model, 'predict_proba'):
                    preds = sub_model.predict_proba(X)[:, 1]
                elif hasattr(sub_model, 'predict'):
                     # Handle models like LightGBM Booster directly
                     preds = sub_model.predict(X)
                     # Ensure output is probability-like if needed
                     if np.max(preds) > 1.0 or np.min(preds) < 0.0:
                           print("Warning: Predictions seem not to be probabilities, applying sigmoid...")
                           preds = 1.0 / (1.0 + np.exp(-preds)) # Sigmoid
                else:
                     raise TypeError(f"Model type {type(sub_model)} has no 'predict_proba' or 'predict' method.")
                all_preds.append(preds)
            except Exception as e:
                print(f"Error predicting with a sub-model: {e}")
                # Handle error: maybe return NaN or skip this model's prediction
                # For now, re-raise or return NaN array
                raise # Or return np.full(X.shape[0], np.nan)
        # Check if any predictions were made
        if not all_preds:
             raise ValueError("No predictions could be generated from the k-fold models.")
        # Average the predictions
        return np.mean(np.array(all_preds), axis=0)

    else: # Handle single model
        print(f"Predicting with single model of type: {type(model)}")
        if hasattr(model, 'predict_proba'):
            # Standard Scikit-learn interface
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, 'predict'):
             # Handle models like LightGBM Booster directly
             preds = model.predict(X)
             # Ensure output is probability-like if needed (e.g., if it's raw scores)
             if np.max(preds) > 1.0 or np.min(preds) < 0.0:
                 print("Warning: Predictions seem not to be probabilities, applying sigmoid...")
                 preds = 1.0 / (1.0 + np.exp(-preds)) # Sigmoid
             return preds
        else:
            raise TypeError(f"Model type {type(model)} has no 'predict_proba' or 'predict' method.")


def evaluate_model(y_true, y_pred_proba, model_name):
    """Calculates and prints classification metrics."""
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        print(f"Warning: Could not calculate AUC for {model_name}. Check if all predictions are the same.")
        auc = np.nan

    print(f"\n--- Evaluation Metrics for {model_name} ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    # print("\nClassification Report:")
    # print(classification_report(y_true, y_pred_binary, zero_division=0))
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_true, y_pred_binary))

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc
    }
    return metrics

def plot_roc_curves(results, save_path):
    """Plots ROC curves for multiple models."""
    plt.figure(figsize=(10, 8))
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_pred_proba'])
        auc = data['metrics']['AUC']
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path)
    print(f"ROC curve plot saved to: {save_path}")
    plt.close()

def plot_metrics_comparison(metrics_df, save_path):
    """Plots a bar chart comparing model metrics."""
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    plot_df = metrics_df.set_index('Model')[metrics_to_plot]

    plot_df.plot(kind='bar', figsize=(14, 8), rot=0)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0.8, 1.0) # Adjust ylim based on typical score range
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Metrics comparison plot saved to: {save_path}")
    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Load Data
    features, labels, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # --- Placeholder for User Confirmation ---
    print("\n--- WAITING FOR USER CONFIRMATION ---")
    print("Please confirm the paths for base models and their metrics in BASE_MODEL_PATHS and BASE_METRICS_PATHS.")
    print("Also confirm the strategy for Stacking OOF predictions and Weighted Averaging weights.")
    print("Script will exit until paths and strategy are confirmed and updated in the code.")
    # In a real scenario, you'd likely get this info and update the dicts above
    # For now, we exit to prevent errors with placeholder paths
    
    # Example check (replace with actual logic based on user input later)
    paths_confirmed = False # Set to True once user confirms and paths are updated
    if not paths_confirmed:
         print("\nPlease update the placeholder paths and configurations in the script before running.")
         exit()
    # --- End Placeholder ---


    # 2. Load Base Models
    base_models = {}
    for name, path in BASE_MODEL_PATHS.items():
        try:
            base_models[name] = load_model(path)
        except Exception as e:
            print(f"Failed to load base model {name} from {path}: {e}")
            # Decide how to handle: skip model, exit, etc.
            exit() # Exit if a base model fails to load

    # 3. Get Base Model Predictions on Test Set
    base_predictions_test = {}
    results_collector = {} # To store true labels and preds for plotting
    all_metrics = []

    print("\nGenerating predictions from base models on the test set...")
    for name, model in base_models.items():
        try:
            preds_test = get_predictions(model, X_test)
            base_predictions_test[name] = preds_test
            metrics = evaluate_model(y_test, preds_test, name)
            all_metrics.append(metrics)
            results_collector[name] = {'y_true': y_test, 'y_pred_proba': preds_test, 'metrics': metrics}
        except Exception as e:
            print(f"Failed to get predictions for model {name}: {e}")
            # Handle appropriately

    # --- Stacking ---
    print("\n--- Stacking Ensemble ---")
    # Strategy Needed: OOF vs Retrain for meta-features
    # Assuming Retrain for now (simpler, needs user confirmation if OOF required)

    # Generate meta-features by predicting on training set
    print("Generating meta-features for Stacking (predicting on training set)...")
    base_predictions_train = {}
    for name, model in base_models.items():
         try:
             # IMPORTANT: If models were k-fold, ideally use OOF preds.
             # If models were trained on full data before, predicting on train set leads to leakage.
             # Safest: Retrain base models here on X_train (or folds of X_train for OOF)
             # Simple approach (potential leakage): Predict directly if model wasn't trained on this exact X_train split
             # Let's assume we predict directly for now, user must be aware of implications
             print(f"Warning: Predicting {name} on training data directly. Ensure model wasn't trained on this exact split to avoid leakage.")
             preds_train = get_predictions(model, X_train)
             base_predictions_train[name] = preds_train
         except Exception as e:
             print(f"Failed to get training predictions for model {name}: {e}")
             # Handle appropriately

    # Check if we got all predictions needed
    if len(base_predictions_train) != len(base_models):
        print("Error: Could not generate training predictions for all base models. Exiting.")
        exit()

    X_meta_train = np.column_stack(list(base_predictions_train.values()))
    X_meta_test = np.column_stack(list(base_predictions_test.values()))

    print(f"Meta-features shape: Train={X_meta_train.shape}, Test={X_meta_test.shape}")

    # Train Meta-Learner (Logistic Regression)
    print("Training Stacking meta-learner (Logistic Regression)...")
    meta_learner = LogisticRegression(random_state=RANDOM_STATE)
    meta_learner.fit(X_meta_train, y_train)

    # Evaluate Stacking Model
    stacking_preds_test = meta_learner.predict_proba(X_meta_test)[:, 1]
    stacking_metrics = evaluate_model(y_test, stacking_preds_test, "Stacking Ensemble")
    all_metrics.append(stacking_metrics)
    results_collector["Stacking Ensemble"] = {'y_true': y_test, 'y_pred_proba': stacking_preds_test, 'metrics': stacking_metrics}


    # --- Weighted Averaging ---
    print("\n--- Weighted Averaging Ensemble ---")

    # Simple Average
    print("Calculating Simple Average Ensemble...")
    simple_avg_preds = np.mean(list(base_predictions_test.values()), axis=0)
    simple_avg_metrics = evaluate_model(y_test, simple_avg_preds, "Simple Average")
    all_metrics.append(simple_avg_metrics)
    results_collector["Simple Average"] = {'y_true': y_test, 'y_pred_proba': simple_avg_preds, 'metrics': simple_avg_metrics}


    # Weighted Average (using base model AUCs)
    print("Calculating Weighted Average Ensemble (based on individual AUC)...")
    base_aucs = {}
    total_auc = 0
    print("Loading base model metrics for weighting...")
    try:
        for name, path in BASE_METRICS_PATHS.items():
            if name in results_collector: # Use AUC calculated on current test set if available
                 auc = results_collector[name]['metrics']['AUC']
                 if pd.isna(auc):
                      print(f"Warning: AUC for {name} is NaN. Excluding from weighted average.")
                      continue
                 base_aucs[name] = auc
                 total_auc += auc
                 print(f"Using AUC for {name}: {auc:.4f}")
            else: # Fallback to loading from file (less ideal if test set differs)
                 print(f"Warning: AUC for {name} not found in current run, loading from {path}")
                 metrics_df = pd.read_csv(path)
                 # Assuming metric names are consistent, find 'auc' or 'AUC'
                 auc_col = 'auc' if 'auc' in metrics_df.columns else 'AUC'
                 if auc_col not in metrics_df.columns:
                      print(f"Error: Cannot find AUC column in {path}. Skipping {name} for weighted avg.")
                      continue
                 # Assuming the relevant metric is the last row or identified uniquely
                 auc = metrics_df.iloc[-1][auc_col] # Adjust if needed
                 if pd.isna(auc):
                      print(f"Warning: Loaded AUC for {name} from {path} is NaN. Excluding.")
                      continue
                 base_aucs[name] = auc
                 total_auc += auc
                 print(f"Loaded AUC for {name} from {path}: {auc:.4f}")

        if total_auc > 0 and len(base_aucs) > 0:
            weighted_preds = np.zeros_like(simple_avg_preds)
            print("Weights:")
            for name, auc in base_aucs.items():
                weight = auc / total_auc
                print(f"  {name}: {weight:.4f} (AUC={auc:.4f})")
                weighted_preds += base_predictions_test[name] * weight

            weighted_avg_metrics = evaluate_model(y_test, weighted_preds, "Weighted Average (AUC)")
            all_metrics.append(weighted_avg_metrics)
            results_collector["Weighted Average (AUC)"] = {'y_true': y_test, 'y_pred_proba': weighted_preds, 'metrics': weighted_avg_metrics}
        else:
            print("Could not calculate weighted average: No valid AUCs found or total AUC is zero.")

    except Exception as e:
        print(f"Error during weighted average calculation: {e}")


    # --- Comparison and Visualization ---
    print("\n--- Results Comparison ---")
    metrics_summary_df = pd.DataFrame(all_metrics)
    print(metrics_summary_df.to_string())

    # Save results
    summary_save_path = os.path.join(SAVE_DIR, 'ensemble_comparison_metrics.csv')
    metrics_summary_df.to_csv(summary_save_path, index=False)
    print(f"Comparison metrics saved to: {summary_save_path}")

    # Generate plots
    roc_save_path = os.path.join(FIGURES_DIR, 'roc_curve_comparison.png')
    plot_roc_curves(results_collector, roc_save_path)

    metrics_plot_save_path = os.path.join(FIGURES_DIR, 'metrics_bar_comparison.png')
    plot_metrics_comparison(metrics_summary_df, metrics_plot_save_path)

    print("\nExperiment 4 finished.") 