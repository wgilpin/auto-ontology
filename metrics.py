import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import sklearn.metrics
from linear_assignment import linear_assignment

nmi = normalized_mutual_info_score
ari = adjusted_rand_score



def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def plot_confusion(y, y_pred, mapping, save_dir: str=None, size: int=8):

    sns.set(font_scale=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)
    entity_label_tups = [(k,v) for k,v in mapping.items()]
    entity_labels = [v for v,v in sorted(entity_label_tups, key=lambda tup: tup[0])]

    # re-order confusion matrix into a diagonal

    for y_hat in range(len(confusion_matrix)):
        max = np.argmax(confusion_matrix[y_hat])
        confusion_matrix.T[[y_hat, max]] = confusion_matrix.T[[max, y_hat]]
    plt.figure(figsize=(size, size))
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap=sns.color_palette("crest", as_cmap=True),
        cbar=False,
        annot_kws={"size": 20},
        xticklabels=entity_labels,
        yticklabels=entity_labels,
        )
    plt.title("Confusion matrix\n", fontsize=20)
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Cluster label', fontsize=20)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()
