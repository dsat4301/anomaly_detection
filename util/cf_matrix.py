import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# noinspection SpellCheckingInspection,SpellCheckingInspection
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          threshold=None):
    """
    Return a plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x, y axis.
                   Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams
    value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    threshold:     Threshold value that leads to the confusion matrix passed.

    percent:       Indicates whether rates in the single categories should be displayed.

    """
    tn, fp, fn, tp = cf.ravel()

    # SWAP CONFUSION MATRIX ORDER
    if cf.shape == (2, 2):
        cf = np.array([[tp, fn], [fp, tn]])

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for _ in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in
                  zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:

        # Accuracy is sum of diagonal divided by total observations
        accuracy = (tp + tn) / (tn + fp + fn + tp)

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * precision * recall / (precision + recall)

            tpr = recall
            tnr = tn / (tn + fp)

            ppv = precision
            npv = tn / (tn + fn)

            informedness = tpr + tnr - 1
            markedness = ppv + npv - 1

            stats_text = f"\n\nAccuracy={accuracy:0.4f}" \
                         f"\nPrecision={precision:0.4f}" \
                         f"\nRecall={recall:0.4f}" \
                         f"\nF1 Score={f1_score:0.4f}" \
                         f"\nInformedness={informedness:.4f}" \
                         f"\nMarkedness={markedness:.4f}"

            if threshold is not None:
                stats_text += f"\nThreshold={threshold:0.4f}"
        else:
            stats_text = f"\n\nAccuracy={accuracy:0.4f}"
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plot = plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar,
                xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('Actual label')
        plt.xlabel('Assigned label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    return plot
