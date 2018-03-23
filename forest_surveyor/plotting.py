import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# helper function for plotting conf mat
def plot_confusion_matrix(cm, class_names=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

# helper plot for viewing differences between feature usage
# combining st errs by division: https://chem.libretexts.org/Core/Analytical_Chemistry/Quantifying_Nature/Significant_Digits/Propagation_of_Error
def log_ratio_plot(num, denom, labels, num_err=None, denom_err=None, top=3):
    fig, ax = plt.subplots(figsize=(11, 3))
    log_ratio = np.log(num/denom)

    top_n = np.flip(np.argpartition(log_ratio, -top)[-top:], axis=0)
    bot_n = np.flip(np.argpartition(-log_ratio, -top)[-top:], axis=0)

    lr_top = [log_ratio[i] for i in top_n]
    lr_bot = [log_ratio[i] for i in bot_n]

    if num_err is not None and denom_err is not None:
        ax.stem(top_n, lr_top, linefmt = 'C' + str(1) + ':', markerfmt = 'C' + str(1) + '.')
        ax.stem(bot_n, lr_bot, linefmt = 'C' + str(2) + ':', markerfmt = 'C' + str(2) + '.')

        yerr = 0.434*np.sqrt((num_err/num)**2 + (denom_err/denom)**2)
        ax.errorbar(range(len(labels)), log_ratio, yerr = yerr, fmt='o')
    else:
        ax.stem(range(len(labels)), log_ratio)
        ax.stem(top_n, lr_top, linefmt = 'C' + str(1) + '-', markerfmt = 'C' + str(1) + 'o')
        ax.stem(bot_n, lr_bot, linefmt = 'C' + str(2) + '-', markerfmt = 'C' + str(2) + 'o')

    ax.axhline(0.0, color = 'k', ls = '--')
    ax.annotate('1:1', xy=(-1.0, max(log_ratio) * 0.1))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel('log(ratio)')

    plt.show()
    if num_err is not None:
        return(log_ratio, yerr)
    else:
        return(log_ratio)
