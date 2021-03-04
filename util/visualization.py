from abc import abstractmethod
from typing import Collection, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

from util.cf_matrix import make_confusion_matrix


# noinspection SpellCheckingInspection
class SaveableFigure:

    @abstractmethod
    def savefig(self, fname, *, transparent=None, **kwargs):
        pass


def get_boxplot(
        scores_data_frame: pd.DataFrame,
        filter_condition: Collection[bool],
        x: str = None,
        anomaly_score_label: str = 'anomaly_score'):
    fig, ax = plt.subplots(figsize=(10, 10))
    boxplot = sns.boxplot(
        data=scores_data_frame.loc[filter_condition],
        x=x,
        y=anomaly_score_label,
        ax=ax)

    return boxplot.get_figure()


def get_displot(
        scores_data_frame: pd.DataFrame,
        filter_condition: Collection[bool],
        hue: str = None,
        anomaly_score_label: str = 'anomaly_score',
        log_scale: bool = True):
    return sns.displot(
        data=scores_data_frame.loc[filter_condition],
        x=anomaly_score_label,
        hue=hue,
        kind='kde',
        log_scale=log_scale,
        height=10,
        rug=True)


def get_confusion_matrix(
        scores_data_frame: pd.DataFrame,
        threshold: float,
        anomaly_class_label: str,
        anomaly_score_label: str = 'anomaly_score'):
    y_true = scores_data_frame[anomaly_class_label]
    y_prediction = (scores_data_frame[anomaly_score_label] >= threshold).astype(int)

    cm_labels = ['True Pos', 'False Neg', 'False Pos', 'True Neg']
    categories = [anomaly_class_label, 'normal']

    return make_confusion_matrix(
        confusion_matrix(y_true, y_prediction),
        group_names=cm_labels,
        categories=categories,
        cmap='binary',
        threshold=threshold)


def get_visualizations(
        test_scores: pd.DataFrame,
        filter_condition: Collection[bool],
        threshold: float,
        anomaly_class_label: str = 'anomaly',
        anomaly_score_label: str = 'anomaly_score',
        log_scale: bool = True) -> Collection[Tuple[str, SaveableFigure]]:
    boxplot = get_boxplot(
        scores_data_frame=test_scores,
        filter_condition=filter_condition,
        anomaly_score_label=anomaly_score_label,
        x=anomaly_class_label)
    displot = get_displot(
        scores_data_frame=test_scores,
        filter_condition=filter_condition,
        anomaly_score_label=anomaly_score_label,
        hue=anomaly_class_label,
        log_scale=log_scale)
    cm = get_confusion_matrix(
        scores_data_frame=test_scores,
        threshold=threshold,
        anomaly_class_label=anomaly_class_label,
        anomaly_score_label=anomaly_score_label)

    return list([
        ('boxplot', boxplot),
        ('displot', displot),
        ('confusion_matrix', cm)])


def log_visualizations(
        figures: Collection[Tuple[str, SaveableFigure]],
        base_path: str,
        run_id: str):
    if base_path[-1] != '/':
        base_path += '/'

    with mlflow.start_run(run_id=run_id):
        for name_figure_tuple in figures:
            file_name = base_path + name_figure_tuple[0] + '.png'
            save_figure(file_name, name_figure_tuple[1])
            mlflow.log_artifact(file_name)


def save_figure(file_name: str, figure: SaveableFigure):
    figure.savefig(file_name, bbox_inches="tight")


def get_scatter_plot(
        data: pd.DataFrame,
        filter_condition: Collection[bool],
        x: str,
        y: str,
        hue: str):
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter_plot = sns.scatterplot(
        data=data.loc[filter_condition],
        hue=hue,
        x=x,
        y=y,
        ax=ax)

    return scatter_plot.get_figure()


def get_max_f1_score_threshold(y_true: np.array, y_score: np.array, values_range: Tuple = None):
    if values_range is None:
        values_range = (y_score.min(), y_score.max())

    scores_in_range = y_score[np.array((y_score > values_range[0]) & (y_score < values_range[1]))]

    # noinspection PyUnresolvedReferences
    f1_score_tpl = np.array([(f1_score(y_true, (y_score >= scores_in_range[i]).astype(int)), int(i))
                             for i in range(len(scores_in_range))])

    max_f1_tpl = sorted(f1_score_tpl, key=lambda x: x[0], reverse=True)[0]

    max_f1 = max_f1_tpl[0]
    threshold = scores_in_range[int(max_f1_tpl[1])]

    return max_f1, threshold
