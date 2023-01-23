import json
from processing import SEQ_LEN, SPECIAL_TOKEN_TO_ID


################################################################################


BATCH_SIZE = 18
EPOCHS = 100

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


def config_json():
    with open("./data/vocabulary.json") as json_file:
        vocab = json.load(json_file)
    json_data = {"vocab_size" : len(vocab["token_to_id"]),
        "batch_size" : BATCH_SIZE, "epochs" : EPOCHS, "n_layers" : N_LAYERS,
        "d_ff" : D_FF, "d_model" : D_MODEL, "d_k" : D_K, "d_v" : D_V,
        "seq_len" : SEQ_LEN, "pad_token_id" : SPECIAL_TOKEN_TO_ID["[PAD]"]}
    with open("config.json", "w") as json_file:
        json.dump(json_data, json_file, indent=2)


################################################################################


def main():
    config_json()

if __name__ == "__main__":
    main()
