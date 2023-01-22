import json
import torch
from data.data_loading import init_dataloader
from gen_config import AttrDict


################################################################################


CONFIG_PATH = "./config.json"
DATASET_PATH = "./data/cards_dataset.json"
MODELS_DIR = "./models/"


################################################################################


def inference(model, batch):
    pass


################################################################################


def main():
    dataloader = init_dataloader(DATASET_PATH, 1)
    for 

    model = torch.load(f"{MODELS_DIR}ImgBert")

if __name__ == "__main__":
    main()
