import argparse
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_dataset(csv_path, sample_size=200_000, random_state=42):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)

    X = df[["Re", "Im", "SNR"]].values
    y = df["Label"].values
    return X, y


def get_models():
    return {
        "SVM": SVC(kernel="rbf", gamma="scale"),
        "RF": RandomForestClassifier(n_estimators=50),
        "DT": DecisionTreeClassifier(),
        "LR": LogisticRegression(max_iter=500),
    }


def train_and_evaluate_models(X, y, models, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    accuracies = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies[name] = accuracy
        trained_models[name] = model
        print(f"{name}: {accuracy:.4f}")

    return accuracies, trained_models, x_test, y_test


def plot_accuracy_comparison(accuracies, output_path):
    names = list(accuracies.keys())
    values = list(accuracies.values())

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("ML Model Accuracy Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_models(models, output_dir):
    for name, model in models.items():
        joblib.dump(model, os.path.join(output_dir, f"{name}_model.pkl"))


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate ML models for OFDM signal detection.")
    parser.add_argument("--dataset", default="../data/dataset.csv", help="Path to the dataset CSV file.")
    parser.add_argument("--sample-size", type=int, default=200_000, help="Number of rows to sample from the dataset.")
    parser.add_argument("--output-dir", default="../results", help="Output directory for plots and models.")
    return parser.parse_args()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args = parse_args()

    dataset_path = os.path.abspath(os.path.join(script_dir, args.dataset))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    X, y = load_dataset(dataset_path, sample_size=args.sample_size)
    models = get_models()
    accuracies, trained_models, x_test, y_test = train_and_evaluate_models(X, y, models)

    # Plot accuracy comparison
    accuracy_plot_path = os.path.join(output_dir, "ml_accuracy.png")
    plot_accuracy_comparison(accuracies, accuracy_plot_path)

    # Find best model and plot its confusion matrix
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = trained_models[best_model_name]
    best_predictions = best_model.predict(x_test)

    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        y_test,
        best_predictions,
        f"Confusion Matrix ({best_model_name})",
        confusion_matrix_path,
    )

    # Plot individual confusion matrices and save models
    for name, model in trained_models.items():
        predictions = model.predict(x_test)
        individual_cm_path = os.path.join(output_dir, f"conf_{name}.png")
        plot_confusion_matrix(
            y_test,
            predictions,
            f"{name} Confusion Matrix",
            individual_cm_path,
        )

    save_models(trained_models, output_dir)

    print(f"Saved accuracy plot to: {accuracy_plot_path}")
    print(f"Saved confusion matrix to: {confusion_matrix_path}")
    print(f"Saved individual confusion matrices and models to: {output_dir}")


if __name__ == "__main__":
    main()
