import json

import mlflow
import numpy as np


def log_model_summary(
        name: str,
        dataset: str,
        anomaly_score_prediction: np.array,
        threshold: float,
        base_path: str,
        roc_auc_score: float,
        run_id: str = None,
        mean_fit_time: float = None,
        n_trainable_parameters: int = None):
    base_path = base_path if base_path[-1] == '/' else base_path + '/'

    data_file_path = base_path + 'data.json'
    scores_file_path = base_path + 'anomaly_scores.npy'

    data_dict = {
        'name': name,
        'dataset': dataset,
        'mean_fit_time': mean_fit_time,
        'n_trainable_parameters': n_trainable_parameters,
        'threshold': threshold,
        'roc_auc_score': roc_auc_score,
    }

    with open(data_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(data_dict, ensure_ascii=True, fp=outfile, indent=4)

    np.save(arr=anomaly_score_prediction, file=scores_file_path)

    if run_id is not None:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(data_file_path)
            mlflow.log_artifact(scores_file_path)
