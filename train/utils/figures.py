import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics

import utils.settings as st


def prediction_distribution(y_pred, set):
    # Create distribution plot
    pd.cut(y_pred, bins=np.arange(0, 1.1, 0.1)
           ).value_counts().plot(kind='bar')
    plt.title(f'{set} score distribution')
    plt.subplots_adjust(left=0.25)
    plt.savefig(st.IMAGES_PATH + f'{set} scores.png')
    plt.close()


def confusion_matrices(set, y_set,
                       y_pred_scores, y_pred_labels):
    # Confusion matrices plots for different thresholds
    for cut_point in st.CUT_POINTS:
        cm = metrics.confusion_matrix(y_set,
                                      (y_pred_scores >= cut_point).astype(int))
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(st.IMAGES_PATH + f'{set} cm {str(cut_point)[2:]}.png')

    # Confusion matrices plots for recommended threshold
    cm = metrics.confusion_matrix(y_set, y_pred_labels)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(st.IMAGES_PATH + f'{set} confusion matrix.png')


def format_plot(title, xlabel, ylabel):
    '''
    Function to add format to plot
    '''
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid('on')
    plt.axis('square')
    plt.ylim((-0.05, 1.05))
    plt.legend()
    plt.tight_layout()
    pass


def roc_curves(labels, prediction_scores, legend,
               title, x_label, y_label,
               color=st.COLORS[1]):
    # ROC AUC
    fpr, tpr, _ = metrics.roc_curve(labels, prediction_scores, pos_label=1)
    auc = metrics.roc_auc_score(labels, prediction_scores)
    legend_string = legend + ' ($AUC = {:0.4f}$)'.format(auc)

    # Create plot
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=legend_string, color=color)

    # Format plot
    format_plot(title, x_label, y_label)

    # Save plot
    plt.savefig(st.IMAGES_PATH + 'roc.png', dpi=150)
    plt.close()


def plot_prc(labels, prediction_scores, legend,
             title, x_label, y_label,
             color=st.COLORS[1]):
    '''
    Function to plot PRC curve
    '''
    precision, \
        recall, \
        _ = metrics.precision_recall_curve(labels,
                                           prediction_scores)
    average_precision = metrics.average_precision_score(
        labels, prediction_scores)
    legend_string = legend + ' ($AP = {:0.4f}$)'.format(average_precision)

    # Create plot1
    plt.plot(recall, precision, label=legend_string, color=color)

    # Format plot
    format_plot(title, x_label, y_label)

    # Save plot
    plt.savefig(st.IMAGES_PATH + 'pcr.png', dpi=150)
    plt.close()


def plot_ks(labels, prediction_scores,
            color=st.COLORS):
    '''
    Function to plot KS plot
    '''
    # KS

    fpr, tpr, \
        thresholds = metrics.roc_curve(labels, prediction_scores,
                                       pos_label=1)
    fnr = 1 - tpr
    tnr = 1 - fpr
    thresholds[0] = 1
    plt.plot(thresholds, fnr, label='FNR (Class 1 Cum. Dist.)',
             color=color[0], lw=1.5)
    plt.plot(thresholds, tnr, label='TNR (Class 0 Cum. Dist.)',
             color=color[1], lw=1.5)

    kss = tnr - fnr
    ks = kss[np.argmax(np.abs(kss))]
    t_ = thresholds[np.argmax(np.abs(kss))]

    # Create plot1
    plt.vlines(t_, tnr[np.argmax(np.abs(kss))], fnr[np.argmax(
        np.abs(kss))], colors='red', linestyles='dashed')

    # Format plot
    format_plot(f'Test KS = {ks}; {t_} Threshold',
                'Threshold', 'Rate (Cumulative Distribution)')

    # Save plot
    plt.savefig(st.IMAGES_PATH + 'ks_curve.png', dpi=150)
    plt.close()


def plot_metrics(y_test,
                 test_y_pred_scores,
                 test_y_pred_labels):
    # Plot distribution
    prediction_distribution(test_y_pred_scores, "Test set")
    # Plot confusion matrix
    confusion_matrices("Test set", y_test,
                       test_y_pred_scores, test_y_pred_labels)
    # Plot roc auc
    roc_curves(y_test, test_y_pred_scores,
               'Test', 'ROC',
               'False Positive Rate',
               'True Positive Rate (Recall)')
    # Plot prc
    # plot_prc(y_test, test_y_pred_scores,
    #          'Test', 'PRC',
    #          'recision', 'Recall')
    # Plot KS
    # plot_ks(y_test, test_y_pred_scores,
    #         ['#fb6376', '#3a7ca5'])
