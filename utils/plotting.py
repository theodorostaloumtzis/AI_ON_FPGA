import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


# confusion matrix code from Maurizio
# /eos/user/m/mpierini/DeepLearning/ML4FPGA/jupyter/HbbTagger_Conv1D.ipynb
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    cbar = plt.colorbar()
    plt.clim(0, 1)
    cbar.set_label(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plotRoc(fpr, tpr, auc, labels, linestyle, legend=True):
    for _i, label in enumerate(labels):
        plt.plot(
            tpr[label],
            fpr[label],
            label='{} tagger, AUC = {:.1f}%'.format(label.replace('j_', ''), auc[label] * 100.0),
            linestyle=linestyle,
        )
    plt.semilogy()
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.001, 1)
    plt.grid(True)
    if legend:
        plt.legend(loc='upper left')
    plt.figtext(0.25, 0.90, 'hls4ml', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)


def rocData(y, predict_test, labels):
    df = pd.DataFrame()

    fpr = {}
    tpr = {}
    auc1 = {}

    for i, label in enumerate(labels):
        df[label] = y[:, i]
        df[label + '_pred'] = predict_test[:, i]

        fpr[label], tpr[label], threshold = roc_curve(df[label], df[label + '_pred'])

        auc1[label] = auc(fpr[label], tpr[label])
    return fpr, tpr, auc1


def makeRoc(y, predict_test, labels, linestyle='-', legend=True):
    if 'j_index' in labels:
        labels.remove('j_index')

    fpr, tpr, auc1 = rocData(y, predict_test, labels)
    plotRoc(fpr, tpr, auc1, labels, linestyle, legend=legend)
    return predict_test


def print_dict(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))
            
            
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def _infer_n_classes(Y):
    # Works for one-hot (N,C) or label vector (N,)
    if Y.ndim == 2:
        return Y.shape[1]
    return int(np.max(Y)) + 1

def _to_label_vector(Y):
    # Converts one-hot to labels; passes labels through unchanged
    return np.argmax(Y, axis=1) if Y.ndim == 2 else Y

def plotROC(Y, y_pred, y_pred_hls4ml, label="Model", zoom=None):
    """
    Y:            (N,C) one-hot OR (N,) integer labels
    y_pred:       (N,C) probabilities/logits for Keras model
    y_pred_hls4ml:(N,C) probabilities/logits for hls4ml model
    label:        text annotation on the figure
    zoom:         None or (xmin, xmax, ymin, ymax) to set axis limits
    """
    # ----- infer class count & labels -----
    n_classes = _infer_n_classes(np.asarray(Y))
    labels = [str(i) for i in range(n_classes)]

    # ----- accuracy (uses argmax; fine for both logits and probs) -----
    y_true  = _to_label_vector(np.asarray(Y))
    yhat_1  = np.argmax(np.asarray(y_pred), axis=1)
    yhat_2  = np.argmax(np.asarray(y_pred_hls4ml), axis=1)
    accuracy_keras  = float(accuracy_score(y_true, yhat_1))
    accuracy_hls4ml = float(accuracy_score(y_true, yhat_2))
    print("Accuracy Keras:  {}".format(accuracy_keras))
    print("Accuracy hls4ml: {}".format(accuracy_hls4ml))

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(9, 9))
    # Keras ROC
    makeRoc(Y, y_pred, labels=labels)
    # Reset color cycle so the per-class colors match between models
    ax.set_prop_cycle(None)
    # hls4ml ROC (dashed)
    makeRoc(Y, y_pred_hls4ml, labels=labels, linestyle='--')

    # Legend that distinguishes model (line style), not classes
    from matplotlib.lines import Line2D
    from matplotlib.legend import Legend
    lines = [Line2D([0], [0], ls='-'), Line2D([0], [0], ls='--')]
    leg = Legend(ax, lines, labels=['Keras', 'hls4ml'], loc='lower right', frameon=False)
    ax.add_artist(leg)

    # Optional zoom; otherwise full ROC square
    if zoom is not None:
        xmin, xmax, ymin, ymax = zoom
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    else:
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

    # Diagonal chance line (nice reference)
    ax.plot([0, 1], [0, 1], ls=":", lw=1)

    # Title/labels and annotation
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC per class")
    plt.figtext(0.18, 0.18, label, wrap=True, ha='left', va='center')
    plt.tight_layout()
    return fig, ax

# Examples (unchanged; add zoom if you still want a tight view):
# fig1, ax1 = plotROC(y_test, y_predict,   y_predict_hls4ml,   label="Keras")
# fig2, ax2 = plotROC(y_test, y_predict_q, y_predict_hls4ml_q, label="QKeras")
# If you want the tight corner:
# fig1, ax1 = plotROC(y_test, y_predict, y_predict_hls4ml, label="Keras", zoom=(0.75,1.0,0.75,1.0))
