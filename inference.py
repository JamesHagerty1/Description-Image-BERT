import torch


################################################################################


MODEL_PATH = "./models/ImgBert-loss:0.021"


################################################################################


def inference(x, y_i, model_path=MODEL_PATH):
    model = torch.load(model_path)
    with torch.no_grad():
        y_hat, attn = model(x, y_i)
    return y_hat, attn


################################################################################


def main():
    pass

if __name__ == "__main__":
    main()
