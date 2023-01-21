import json
import os
from PIL import Image


################################################################################


DATA_DIR = "data/"
IMG_PIXELS = 4096
BIN_WORD_LEN = 8 
BATCH_SIZE = 18

# Model parameters independent of data
N_LAYERS = 1 
D_FF = D_MODEL = 64 
D_K = D_V = 16 


################################################################################


# More concise syntax to access JSON fields
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def config_json(vocab_size):
    json_data = {
        "vocab_size" : vocab_size,
        "batch_size" : BATCH_SIZE,
        "n_layers" : N_LAYERS,
        "d_ff" : D_FF,
        "d_model" : D_MODEL,
        "d_k" : D_K,
        "d_v" : D_V,}
    with open("config.json", "w") as json_file:
        json.dump(json_data, json_file, indent=2)


################################################################################


def main():
    pass

if __name__ == "__main__":
    main()
