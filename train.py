import json
import torch
import torch.nn as nn
import torch.optim as optim
from data.data_loading import init_dataloader
from model import BERT
from gen_config import AttrDict


################################################################################


CONFIG_PATH = "./config.json"
DATASET_PATH = "./data/cards_dataset.json"
MODELS_DIR = "./models/"


################################################################################


def main():
    with open(CONFIG_PATH) as json_file:
        json_data = json_file.read()
    config = json.loads(json_data) 
    c = AttrDict(config) # config, concise JSON object access syntax

    model = BERT(c)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters())
    dataloader = init_dataloader(DATASET_PATH, c.batch_size)
    epochs = c.epochs

    # for epoch in range(epcohs):
    for epoch in range(100000000000):
        loss_sum, iters = 0, 0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            # x: (batch_size, seq_len)
            # y_i: (batch_size, desc_max_masks)
            # y: (batch_size, desc_max_masks)
            x, y_i, y = batch
            # y_hat: (batch_size, desc_max_masks, vocab_size)
            y_hat, _ = model(x, y_i)
            # y_hat_T: (batch_size, vocab_size, desc_max_masks)
            y_hat_T = y_hat.transpose(1, 2)
            loss = criterion(y_hat_T, y).float().mean()
            loss.backward()
            optimizer.step()
            loss_sum, iters = loss_sum + loss.item(), iters + 1
        avg_loss = loss_sum / iters
        print(avg_loss)
        if avg_loss < 0.02:
            torch.save(model, f"{MODELS_DIR}ImgBert-loss:{loss:.2}")
            break

if __name__ == "__main__":
    main()
