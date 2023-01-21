import os
from preprocessing import standardize_image


################################################################################


IMAGES_DIR = "./cards/"
DATASET_DIR = "./cards_dataset/"


################################################################################


def main():
    for filename in os.listdir(IMAGES_DIR):
        input_path = f"{IMAGES_DIR}{filename}"
        output_path = f"{DATASET_DIR}{filename}"
        standardize_image(input_path, output_path)

if __name__ == "__main__":
    main()
