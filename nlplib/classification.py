from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import plot_tree
from nlplib.log import write_log


def check_logistic_regression_prerequisites(
    split: dict[str, list[str]],
    representations: dict,
) -> bool:
    labels = split["train_labels"] + split["test_labels"]
    classes = set(labels)
    if not split["train_texts"] or not split["test_texts"]:
        raise ValueError("Logistic regression requires non-empty train and test sets")
    all_texts = split["train_texts"] + split["test_texts"]
    if len(all_texts) != len(set(all_texts)):
        raise ValueError(
            "Independent cases require every text sample to identify one case"
        )
    if set(split["train_labels"]) != classes:
        if len(classes) >= 2:
            raise ValueError("Every emotion class must be represented in the training set")

    representation_sets = (
        ("frequency", representations["tfidf_train"], representations["tfidf_test"]),
        ("static", representations["static_train"], representations["static_test"]),
        ("contextual", representations["contextual_train"], representations["contextual_test"]),
    )
    for model_type, train_features, test_features in representation_sets:
        train_shape = train_features.shape
        test_shape = test_features.shape
        if train_shape[0] != len(split["train_labels"]):
            raise ValueError(f"{model_type} train rows do not match train labels")
        if test_shape[0] != len(split["test_labels"]):
            raise ValueError(f"{model_type} test rows do not match test labels")
        if train_shape[1] != test_shape[1]:
            raise ValueError(f"{model_type} train/test feature dimensions differ")
        train_values = (
            train_features.data
            if issparse(train_features)
            else np.asarray(train_features)
        )
        test_values = (
            test_features.data
            if issparse(test_features)
            else np.asarray(test_features)
        )
        if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
            raise ValueError(f"{model_type} contains non-finite feature values")

    write_log(
        "Logistic regression prerequisites passed: "
        f"{len(classes)} categorical emotion classes from an exponential-family "
        "multinomial response; independent cases; finite features; matching "
        "dimensions; no normality or homogeneity-of-variance requirement"
    )
    write_log(
        "Logistic regression assumptions: the model uses a linear predictor "
        "through the link function; nonlinear feature transformations are "
        "allowed; errors need not be normal; coefficients use MLE"
    )
    if len(classes) < 2:
        write_log(
            "Emotion classification skipped: a single emotion class cannot be "
            "fit by Logistic Regression or Random Forest"
        )
        return False
    return True


def run_emotion_classification(
    split: dict[str, list[str]],
    representations: dict,
    output_file: str = "data/output/confusion_matrices.png",
) -> dict[str, dict[str, float]]:
    if not check_logistic_regression_prerequisites(split, representations):
        return {}
    labels = sorted(set(split["train_labels"] + split["test_labels"]))
    representation_sets = (
        ("frequency", representations["tfidf_train"], representations["tfidf_test"]),
        ("static", representations["static_train"], representations["static_test"]),
        ("contextual", representations["contextual_train"], representations["contextual_test"]),
    )
    results: dict[str, dict[str, float]] = {}
    matrices: list[tuple[str, str, np.ndarray]] = []

    for representation_name, train_features, test_features in representation_sets:
        logistic_classifier = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        logistic_classifier.fit(train_features, split["train_labels"])
        logistic_predictions = logistic_classifier.predict(test_features)
        logistic_probabilities = logistic_classifier.predict_proba(test_features)
        logistic_metrics = _classification_metrics(
            split["test_labels"],
            logistic_predictions,
            labels,
        )
        results[f"logistic_{representation_name}"] = logistic_metrics
        matrices.append((
            "Logistic regression",
            representation_name,
            confusion_matrix(split["test_labels"], logistic_predictions, labels=labels),
        ))
        _log_confusion_errors(
            f"{representation_name} logistic regression",
            split["test_labels"],
            logistic_predictions,
            labels,
        )
        write_log(
            f"{representation_name} logistic regression statistics: "
            f"classes={len(labels)}, train_rows={train_features.shape[0]}, "
            f"test_rows={test_features.shape[0]}, features={train_features.shape[1]}, "
            f"iterations={logistic_classifier.n_iter_.tolist()}, "
            f"intercept={logistic_classifier.intercept_.tolist()}, "
            f"coefficient_shape={logistic_classifier.coef_.shape}"
        )
        _log_metrics(f"{representation_name} logistic regression", logistic_metrics)
        _plot_logistic_cumulative_distribution(
            logistic_probabilities,
            logistic_classifier.classes_.tolist(),
            representation_name,
        )
        write_log(
            f"{representation_name} logistic regression decision rule: "
            "classify each object as argmax_k P(class_k | x), "
            "where probabilities use the multiclass softmax link"
        )

        forest = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        forest.fit(train_features, split["train_labels"])
        _plot_random_forest_tree(
            forest,
            train_features,
            split["train_labels"],
            representation_name,
        )
        forest_predictions = forest.predict(test_features)
        forest_metrics = _classification_metrics(
            split["test_labels"],
            forest_predictions,
            labels,
        )
        results[f"random_forest_{representation_name}"] = forest_metrics
        matrices.append((
            "Random forest",
            representation_name,
            confusion_matrix(split["test_labels"], forest_predictions, labels=labels),
        ))
        _log_confusion_errors(
            f"{representation_name} random forest",
            split["test_labels"],
            forest_predictions,
            labels,
        )
        write_log(
            f"{representation_name} random forest statistics: "
            f"classes={len(labels)}, train_rows={train_features.shape[0]}, "
            f"test_rows={test_features.shape[0]}, features={train_features.shape[1]}, "
            f"trees={forest.n_estimators}, max_depth={forest.max_depth}, "
            f"max_features={forest.max_features}, "
            f"min_samples_leaf={forest.min_samples_leaf}, "
            f"oob_score={forest.oob_score}, "
            f"feature_importance_mean={forest.feature_importances_.mean():.6f}, "
            f"feature_importance_max={forest.feature_importances_.max():.6f}"
        )
        _log_metrics(f"{representation_name} random forest", forest_metrics)

    figure, axes = plt.subplots(2, len(representation_sets), figsize=(18, 10))
    for axis, (classifier_name, model_type, matrix) in zip(axes.flat, matrices):
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{classifier_name}: {model_type}")
        axis.set_xlabel("Predicted emotion")
        axis.set_ylabel("Actual emotion")
        axis.set_xticks(range(len(labels)), labels)
        axis.set_yticks(range(len(labels)), labels)
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(column, row, matrix[row, column], ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle("Classifier confusion matrices")
    figure.tight_layout()
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    write_log("Classification comparison table:")
    for model_name, metrics in results.items():
        write_log(
            f"{model_name}: "
            + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        )
    return results


def _classification_metrics(
    actual: list[str],
    predicted: list[str],
    labels: list[str],
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, labels=labels, average="macro", zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, labels=labels, average="macro", zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, labels=labels, average="macro", zero_division=0
        ),
    }


def _log_metrics(model_name: str, metrics: dict[str, float]) -> None:
    write_log(
        f"{model_name} evaluation statistics: "
        + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    )


def _log_confusion_errors(
    model_name: str,
    actual: list[str],
    predicted: list[str],
    labels: list[str],
) -> None:
    matrix = confusion_matrix(actual, predicted, labels=labels)
    errors = [
        (matrix[row, column], labels[row], labels[column])
        for row in range(len(labels))
        for column in range(len(labels))
        if row != column and matrix[row, column] > 0
    ]
    errors.sort(reverse=True)
    write_log(
        f"{model_name} most common classification errors: "
        + (
            ", ".join(f"{actual_label}->{predicted_label}={count}" for count, actual_label, predicted_label in errors[:5])
            if errors
            else "none"
        )
    )


def _plot_logistic_cumulative_distribution(
    probabilities: np.ndarray,
    labels: list[str],
    representation_name: str,
    output_directory: str = "data/output/logistic_cdf",
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    cumulative_probabilities = np.cumsum(
        np.sort(probabilities, axis=1),
        axis=1,
    )
    x_values = np.arange(1, cumulative_probabilities.shape[1] + 1)

    figure, axes = plt.subplots(figsize=(9, 6))
    for row in cumulative_probabilities:
        axes.step(x_values, row, where="mid", alpha=0.15, color="steelblue")
    axes.plot(
        x_values,
        cumulative_probabilities.mean(axis=0),
        marker="o",
        color="darkblue",
        label="Mean cumulative probability",
    )
    axes.set_title(f"Logistic regression cumulative distribution: {representation_name}")
    axes.set_xlabel("Number of highest-probability classes included")
    axes.set_ylabel("Cumulative predicted probability")
    axes.set_xticks(x_values, [f"Rank {rank}" for rank in x_values])
    axes.set_ylim(0, 1.05)
    axes.grid(axis="y", alpha=0.25)
    axes.legend()
    figure.tight_layout()
    figure.savefig(
        output_path / f"logistic_cdf_{representation_name}.png",
        dpi=150,
    )
    plt.close(figure)


def _plot_random_forest_tree(
    forest: RandomForestClassifier,
    train_features,
    train_labels: list[str],
    representation_name: str,
    output_directory: str = "data/output/random_forest",
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    svg_path = output_path / f"random_forest_{representation_name}.svg"
    feature_names = [
        f"feature_{index}" for index in range(train_features.shape[1])
    ]
    figure = plt.figure(figsize=(24, 16))
    plot_tree(
        forest.estimators_[0],
        max_depth=5,
        feature_names=feature_names,
        class_names=[str(class_name) for class_name in forest.classes_],
        filled=True,
        rounded=True,
        impurity=False,
    )
    figure.suptitle(f"Random forest tree: {representation_name}")
    figure.tight_layout()
    figure.savefig(svg_path, format="svg")
    plt.close(figure)
    write_log(
        f"Random forest visualization saved: {svg_path} "
        "(first tree, displayed to depth 5)"
    )
