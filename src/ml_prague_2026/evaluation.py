import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from dataclasses import dataclass
from IPython.display import display
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score,
)


@dataclass
class EvaluationMetrics:
    model_name: str
    classification_report: dict
    metrics: pd.DataFrame


def recall_at_k(y_true, y_pred_score):
    k = int(sum(y_true))

    idx = np.argsort(list(y_pred_score))[::-1].copy()
    top_k_idx = idx[:k]

    return float(sum(y_true[top_k_idx]) / sum(y_true))


def evaluate_model(
    model_name: str,
    y_true: List[float],
    y_pred: List[int],
    y_pred_score: List[float],
    show_classification_metrics: List[str] = ["Precision", "Recall"],
    show_ranking_metrics: List[str] = ["AUPRC", "Rec@K"],
    average: str = 'macro avg',
):
    """Evaluate single model and save metrics."""
    classification_metrics = classification_report(y_true, y_pred, output_dict=True)

    ranking_metrics = pd.DataFrame({
        'AUPRC': [average_precision_score(y_true, y_pred_score)],
        'AUC': [roc_auc_score(y_true, y_pred_score)],
        'Rec@K': [recall_at_k(y_true, y_pred_score)],
    })

    metrics = {}
    if show_classification_metrics is not None:
        for m in show_classification_metrics:
            metrics[m] = classification_metrics[average][m.lower()]

    if show_ranking_metrics is not None:
        for m in show_ranking_metrics:
            metrics[m] = ranking_metrics[m]

    if len(metrics) > 0:
        display(pd.DataFrame(metrics).round(3))

    return EvaluationMetrics(
        model_name, classification_metrics, ranking_metrics
    )


def compare_models(
    eval_metrics_list: List[EvaluationMetrics],
    metrics: List[str] = ["Precision", "Recall", "AUPRC", "Rec@K"],
    average: str = 'macro avg',
    show_table: bool = True,
    show_plot: bool = True,
    figsize=None,
):
    """Compare multiple models at once."""
    rows = []

    for em in eval_metrics_list:
        row = {'Model': em.model_name}

        if isinstance(em.metrics, pd.DataFrame) and not em.metrics.empty:
            for col in ['AUPRC', 'AUC', 'Rec@K']:
                if col in em.metrics.columns:
                    row[col] = float(em.metrics.iloc[0][col])

        cr = em.classification_report or {}
        row['Accuracy'] = float(cr.get('accuracy')) if cr.get('accuracy') is not None else np.nan

        avg_block = cr.get(average, {}) if isinstance(cr.get(average, {}), dict) else {}
        row['Precision'] = float(avg_block.get('precision')) if avg_block.get('precision') is not None else np.nan
        row['Recall'] = float(avg_block.get('recall')) if avg_block.get('recall') is not None else np.nan
        row['F1'] = float(avg_block.get('f1-score')) if avg_block.get('f1-score') is not None else np.nan

        rows.append(row)

    if not rows:
        empty = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1', 'AUPRC', 'AUC', 'Rec@K'])
        return empty, None, None

    comparison_df = pd.DataFrame(rows).set_index('Model')

    for c in metrics:
        if c not in comparison_df.columns:
            comparison_df[c] = np.nan
    comparison_df = comparison_df[metrics]

    if show_table:
        display(comparison_df.style.format("{:.3f}").highlight_max(axis=0, color='#008F39'))

    fig, ax = None, None
    if show_plot:
        if figsize is None:
            figsize = (2 + len(eval_metrics_list) * 1.5, 3.5)

        models = comparison_df.index.tolist()
        values = comparison_df[metrics].to_numpy(dtype=float)

        x = np.arange(len(models))
        n_metrics = len(metrics)
        width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=figsize)

        for i, metric in enumerate(metrics):
            ax.bar(
                x + (i - (n_metrics - 1) / 2) * width,
                values[:, i],
                width,
                label=metric,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model comparison across evaluation metrics")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

    if not show_table and not show_plot:
        return comparison_df, fig, ax
