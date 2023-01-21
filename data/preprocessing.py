"""
-Helpers to create any image dataset for this model
-60x60 images where the only pixel colors are white, mid-gray, black
"""


from PIL import Image as im


################################################################################


def crop():
    pass


def standardize_image(input_path, output_path):
    print(input_path, output_path)
    image = im.open(input_path)


################################################################################


def main():
    pass

if __name__ == "__main__":
    main()
