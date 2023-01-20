import json
from torch.utils.data import Dataset, DataLoader


################################################################################


TRAIN_FILE = "train.json"


######## Card images as sentences dataset ###################################### 


class CardSentenceDataset(Dataset):
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
                    self.masked_ids.append(
                        eval(masked_entry["masked_tokens_ids"]))
                    self.masked_i.append(eval(masked_entry["masked_tokens_i"]))

    def __getitem__(self, index):
        return self.sentences[index], self.masked_ids[index], \
            self.masked_i[index]

    def __len__(self):
        return len(self.sentences)


######## Card images as sentences dataloader ###################################


def init_dataloader(batch_size):
    dataset = CardSentenceDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
