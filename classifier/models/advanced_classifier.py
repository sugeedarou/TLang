import torch
import pandas as pd
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def edge_detection_after_sort(data, sort_model_predictions, edge_detection_model=None, device=device):
    '''
        sorts 'data' according to 'sort_model_predictions' (entries with same predictions get close)
        then classifies the sorted dataset with 'edge_detection_model'
        -- for a batch_size sensitive edge_detection_classifier 'data' must be a integer multiple of the batch_size

    :param data: a batch of data (preprocessed) to be classified by language
        shape 1) or 2):
        1) (ids, lengths, labels, texts)
        2) (ids, texts)
    :param sort_model_predictions: np.array(len(data)), tensor(length=data) or list()
        the predictions of the sort model, must be aligned with the ids (data[0])
    :param edge_detection_model: class(nn.Module)
        the model that shall be used for edge detected in data accordingly to 'sort_model_predictions'

    :return: np.array(num data_points)
        (edge_detection_model) predictions[i] for data[i]
    '''
    try:
        edge_detection_model.eval()
    except:
        pass
    edge_detection_model.to(device)

    if len(data) == 4:
        ids, length, labels, texts = data
    elif len(data) == 2:
        ids, texts = data
    else:
        print("edge_detection_after_sort got a non-supported format for data, "
              "\n - either use (ids, texts) or (ids, lengths, labels, texts")
        exit(1)

    entry_nr = torch.tensor(range(len(ids)))
    df = pd.DataFrame(list(zip(entry_nr, ids, texts, sort_model_predictions)), columns=["#", "id", "text", "preds"])
    df = df.sort_values(['preds'], ascending=[True])
    df = df.drop("preds", axis=1)

    text_tensor = df["text"].to_numpy()
    text_tensor = torch.stack(list(text_tensor)).to(device)

    edge_det_preds = edge_detection_model.forward(text_tensor)
    edge_det_preds = edge_det_preds.argmax(1)
    edge_det_preds = edge_det_preds.cpu().detach().numpy()

    df["preds"] = edge_det_preds
    df = df.sort_values(['#'], ascending=[True])  # sort it back to the original form

    return df["preds"].to_numpy()

class ensemble_classifier(nn.Module):
    def __init__(self, output_size, n_classifiers):
        super().__init__()
        self.output_size = output_size
        self.n_clfers = n_classifiers
        self.fc = nn.Linear(self.n_clfers, self.output_size)
    def forward(self, classifier_prediction_list):
        out = self.fc(classifier_prediction_list)
        return out
