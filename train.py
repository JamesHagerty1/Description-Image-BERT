from dataloader import init_dataloader
# from model import BERT


def main():
    # I want a metadata json now and dataloader needs to know batch size
    dataloader = init_dataloader()
    # model = BERT()

if __name__ == "__main__":
    main()
