import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from twitter_dataset import TwitterDataset


ef show_confusion_matrix(confmat, save_path="", save_name="confmat.pdf"):
    '''
        plots (and saves) a confusion matrix as combination of numbers and a heatmap
    :param confmat: the confusion matrix
    :param save_path: path for save 
        - default: "" (= save in same directory as main)
    :param save_name: name for the matrix to be saved as (without ending)
            if None: it doesnt get saved
        - default="confmat.pdf"
    '''
    plt.figure(figsize=(15, 15))
    class_names = TwitterDataset.class_names
    df_cm = pd.DataFrame(confmat, index=class_names, columns=class_names).astype(int)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    heatmap = sns.heatmap(df_cm, annot=True, cbar=False, fmt="d", cmap=cmap)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if type(save_name) == str:
        plt.savefig(save_path+save_name+".pdf", dpi=800)
    plt.show()
