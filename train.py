import json
from data.data_loading import init_dataloader
from model import BERT
from gen_config import AttrDict


################################################################################


CONFIG_PATH = "./config.json"
DATASET_PATH = "./data/cards_dataset.json"


################################################################################


def main():
    with open(CONFIG_PATH) as json_file:
        json_data = json_file.read()
    config = json.loads(json_data) 
    c = AttrDict(config) # config, concise JSON object access syntax
    print(c)

    model = BERT(c)
    dataloader = init_dataloader(DATASET_PATH, c.batch_size)

    for i, batch in enumerate(dataloader):
        x, y_i, y = batch
        y_hat = model(x, y_i)
        break
        

if __name__ == "__main__":
    main()
