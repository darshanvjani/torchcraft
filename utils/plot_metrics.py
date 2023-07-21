import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_metrics(exp_metrics):
    sns.set(font_scale=1)
    fig, axs = plt.subplots(2,2,figsize=(25,15))
    plt.rcParams["figure.figsize"] = (25,6)
    train_accuracy,train_losses,test_accuracy,test_losses  = exp_metrics
    
    
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].set_title("Test Accuracy")

    axs[0, 0].plot(train_losses, label="Training Loss")
    axs[0,0].set_xlabel('epochs')
    axs[0,0].set_ylabel('loss')

    axs[1, 0].plot(train_accuracy, label="Training Accuracy")
    axs[1,0].set_xlabel('epochs')
    axs[1,0].set_ylabel('accuracy')

    axs[0, 1].plot(test_losses, label="Validation Loss")
    axs[0,1].set_xlabel('epochs')
    axs[0,1].set_ylabel('loss')

    axs[1, 1].plot(test_accuracy, label="Validation Accuracy")
    axs[1,1].set_xlabel('epochs')
    axs[1,1].set_ylabel('accuracy')
        
    
def plot_confusion_matrix(conf_mat,
                          hide_spines=False,
                          hide_ticks=False,
                          figsize=None,
                          cmap=None,
                          colorbar=False,
                          show_absolute=True,
                          show_normed=False,
                          class_names=None):

    if not (show_absolute or show_normed):
        raise AssertionError('Both show_absolute and show_normed are False')
    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError('len(class_names) should be equal to number of'
                             'classes in the dataset')

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float') / total_samples

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    if figsize is None:
        figsize = (len(conf_mat)*1.25, len(conf_mat)*1.25)

    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                cell_text += format(conf_mat[i, j], 'd')
                if show_normed:
                    cell_text += "\n" + '('
                    cell_text += format(normed_conf_mat[i, j], '.2f') + ')'
            else:
                cell_text += format(normed_conf_mat[i, j], '.2f')
            ax.text(x=j,
                    y=i,
                    s=cell_text,
                    va='center',
                    ha='center',
                    color="white" if normed_conf_mat[i, j] > 0.5 else "black")
    
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
        
    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()