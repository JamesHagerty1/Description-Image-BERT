from torch.utils.data import Dataset, DataLoader


######## Card images as sentences dataset ###################################### 


class CardSentenceDataset(Dataset):
    def __init__(self):
        self.masked_sentences = [42]

    def __index__(self, index):
        return self.masked_sentences[index]

    def __len__(self):
        return len(self.masked_sentences)


################################################################################


def main():
    dset = CardSentenceDataset()
    print('hi')

if __name__ == "__main__":
    main()