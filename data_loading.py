import json
import torch
from torch.utils.data import Dataset, DataLoader


################################################################################


class DescriptionImageDataset(Dataset):
    def __init__(self, path):
        masked_tokens_ids, masked_indices, masked_ids, descriptions = \
            [], [], [], []
        with open(path) as json_file:
            json_data = json.load(json_file)        
        for entry in json_data:
            masked_tokens_ids.append(eval(entry["masked_tokens_ids"]))
            masked_indices.append(eval(entry["masked_indices"]))
            masked_ids.append(eval(entry["masked_ids"]))
            description = []
            tokens, i = eval(entry["tokens"]), 1
            while (tokens[i] not in ["[PAD]", "[IMG]"]):
                description.append(tokens[i])
                i += 1
            descriptions.append(" ".join(description))
        self.masked_tokens_ids = torch.LongTensor(masked_tokens_ids)
        self.masked_indices = torch.LongTensor(masked_indices)
        self.masked_ids = torch.LongTensor(masked_ids)
        self.descriptions = descriptions

    def __getitem__(self, i):
        return self.masked_tokens_ids[i], self.masked_indices[i], \
            self.masked_ids[i], self.descriptions[i]

    def __len__(self):
        return len(self.masked_tokens_ids)


################################################################################


def init_dataloader(dataset_path, batch_size):
    dataset = DescriptionImageDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


################################################################################


def main():
    dataset = DescriptionImageDataset("./cards_dataset.json")
    x, y, y_i, desc = dataset.__getitem__(0)
    print(x.shape, y.shape, y_i.shape)
    print(x[:20])
    print(y)
    print(y_i)
    print(desc)

if __name__ == "__main__":
    main()
