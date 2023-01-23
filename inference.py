import torch
from data_loading import init_dataloader


################################################################################


DATASET_PATH = "./data/cards_dataset.json"
MODELS_DIR = "./models/"


################################################################################


def inference(model, batch):
    with torch.no_grad():
        pass
    pass


################################################################################


def main():
    model = torch.load(f"{MODELS_DIR}ImgBert-loss:0.021")
    dataloader = init_dataloader(DATASET_PATH, 1)

    for i, batch in enumerate(dataloader):
        x, y_i, y, desc = batch

if __name__ == "__main__":
    main()
