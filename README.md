# anomaly_detection

This repository provides implementations of anomaly detection approaches as  [scikit-learn](https://scikit-learn.org/stable/#) estimators. For the neural network-based models, [PyTorch](https://pytorch.org/) is also used. All implement the abstract classes *BaseEstimator* and *OutlierMixin*. The following methods acc. to the scikit-learn API are supported:

* fit
* predict
* fit_predict
* score_samples
* decision_function
* score

For more information, please refer to the scikit-learn [documentation](https://scikit-learn.org/stable/developers/develop.html).

## Installation using conda

``` bash
cd <path-to-anomaly_detection-directory>
conda env create --file anomaly_detection.yml
conda activate anomaly_detection

# test installation
python -m pytest
```

## Content

| Approach             | Estimator                                                                                                       | Reference                                                                                                                                                          |
|----------------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Mahalanobis distance | [MahalanobisDistanceAnomalyDetector](anomaly_detectors/distance_based/mahalanobis_distance_anomaly_detector.py) |                                                                                                                                                                    |
| Euclidean distance   | [EuclideanDistanceAnomalyDetector](anomaly_detectors/distance_based/euclidean_distance_anomaly_detector.py)     |                                                                                                                                                                    |
| Chi-Squared distance | [ChiSquaredDistanceAnomalyDetector](anomaly_detectors/distance_based/chi_squared_distance_anomaly_detector.py)  |                                                                                                                                                                    |
| DeepSVDD             | [DeepSVDDAnomalyDetector](anomaly_detectors/DeepSVDD/SVDD_anomaly_detector.py)                                  | [Ruff, L. et al. Deep one-class classification](http://proceedings.mlr.press/v80/ruff18a.html)                                                                     |
| Autoencoder          | [AEAnomalyDetector](anomaly_detectors/autoencoder/AE_anomaly_detector.py)                                       |                                                                                                                                                                    |
| VAE                  | [VAEAnomalyDetector](anomaly_detectors/variational_autoencoder/VAE_anomaly_detector.py)                         | [Kingma, D. P. & Welling, M. Auto-encoding variational bayes](https://arxiv.org/pdf/1312.6114v10.pdf)                                                              |
| GANomaly             | [GANomalyAnomalyDetector](anomaly_detectors/GANomaly/GANomaly_anomaly_detector.py)                              | [Akcay, S., Atapour-Abarghouei, A. & Breckon, T. P. Ganomaly: Semi-supervised anomaly detection via adversarial training](https://github.com/samet-akcay/ganomaly) |

## Running experiments

For running experiments on any dataset, please refer to the scikit-learn [documentation](https://scikit-learn.org/stable/tutorial/basic/tutorial.html).

## Licence

MIT
