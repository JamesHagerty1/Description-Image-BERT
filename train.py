import json
from data import init_dataloader
from model import BERT
from prepare import AttrDict


################################################################################


CONFIG_FILE = "config.json"


################################################################################


def main():
    with open(CONFIG_FILE) as json_file:
        json_data = json_file.read()
    config = json.loads(json_data) 
    c = AttrDict(config) # config, concise JSON object access syntax
    
    model = BERT(c)
    dataloader = init_dataloader(c.batch_size)

    for i, batch in enumerate(dataloader):
        x, y, y_i = batch
        y_hat = model(x, y_i)
        break
        

if __name__ == "__main__":
    main()
