import json
from data import init_dataloader
# from model import BERT


################################################################################


CONFIG_FILE = "config.json"


################################################################################


def main():
    with open(CONFIG_FILE) as json_file:
        json_data = json_file.read()
    config = json.loads(json_data)
    
    dataloader = init_dataloader(config["batch_size"])
    
    for i, batch in enumerate(dataloader):
        x, y, y_i = batch
        print(x.shape, y.shape, y_i.shape)
        

        break

if __name__ == "__main__":
    main()
