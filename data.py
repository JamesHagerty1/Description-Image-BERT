import json
import torch
from torch.utils.data import Dataset, DataLoader


################################################################################


TRAIN_FILE = "train.json"


######## Card images as sentences dataset ###################################### 


class ImageSentenceDataset(Dataset):
    def __init__(self):
        global TRAIN_FILE
        self.sentences = []
        self.masked_ids = []
        self.masked_i = []
        with open(TRAIN_FILE) as json_file:
            json_data = json.load(json_file)
        for card_entry in json_data:
            for masked_entry in card_entry["masked_sentences"]:
                self.sentences.append(eval(masked_entry["masked_tokens"]))
                self.masked_ids.append(eval(masked_entry["masked_tokens_ids"]))
                self.masked_i.append(eval(masked_entry["masked_tokens_i"]))
        self.sentences = torch.LongTensor(self.sentences)
        self.masked_ids = torch.LongTensor(self.masked_ids)
        self.masked_i = torch.LongTensor(self.masked_i)

    def __getitem__(self, i):
        return self.sentences[i], self.masked_ids[i], self.masked_i[i]

    def __len__(self):
        return len(self.sentences)


######## Card images as sentences dataloader ###################################


def init_dataloader(batch_size):
    dataset = ImageSentenceDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


################################################################################


def main():
    dataset = ImageSentenceDataset()
    x, y, y_i = dataset.__getitem__(0)
    print(x.shape, y.shape, y_i.shape)


if __name__ == "__main__":
    main()