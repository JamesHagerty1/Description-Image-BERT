import json
import torch
from torch.utils.data import Dataset, DataLoader


######## Card images as sentences dataset ###################################### 


class ImageSentenceDataset(Dataset):
    def __init__(self, path):
        masked_tokens_ids, masked_indices, masked_ids = [], [], []
        with open(path) as json_file:
            json_data = json.load(json_file)        
        for entry in json_data:
            masked_tokens_ids.append(eval(entry["masked_tokens_ids"]))
            masked_indices.append(eval(entry["masked_indices"]))
            masked_ids.append(eval(entry["masked_ids"]))
        self.masked_tokens_ids = torch.LongTensor(masked_tokens_ids)
        self.masked_indices = torch.LongTensor(masked_indices)
        self.masked_ids = torch.LongTensor(masked_ids)

    def __getitem__(self, i):
        return self.masked_tokens_ids[i], self.masked_indices[i], \
            self.masked_ids[i]

    def __len__(self):
        return len(self.masked_tokens_ids)


######## Card images as sentences dataloader ###################################


def init_dataloader(batch_size):
    dataset = ImageSentenceDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


################################################################################


def main():
    dataset = ImageSentenceDataset("./cards_dataset.json")
    x, y, y_i = dataset.__getitem__(0)
    print(x.shape, y.shape, y_i.shape)
    print(x[:20])
    print(y)
    print(y_i)

if __name__ == "__main__":
    main()
