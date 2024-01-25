import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, features, labels):
        self.encodings = encodings
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['features'] = torch.FloatTensor(self.features[idx])
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)