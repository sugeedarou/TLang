import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from classifier.twitter_dataset import TwitterDataset


def show_confusion_matrix(confmat):
    plt.figure(figsize=(15,15))
    class_names = TwitterDataset.class_names
    df_cm = pd.DataFrame(confmat, index=class_names, columns=class_names).astype(int)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    heatmap = sns.heatmap(df_cm, annot=True, cbar=False, fmt="d", cmap=cmap)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()