"""
-Helpers to create any image dataset for this model
-60x60 images where the only pixel colors are white, mid-gray, black
"""


from PIL import Image as im


################################################################################


IMG_DIM = 60
# (grayscale) 0-85 -> black, 85-170 -> mid-gray, 170-255-> white
BLACK_MAX = 85
WHITE_MIN = 170


################################################################################


def crop():
    pass


def standardize_image(input_path, output_path):
    def trinary_brightness(brightness):
        if (brightness <= BLACK_MAX): return 0
        elif (BLACK_MAX < brightness < WHITE_MIN): return 128
        else: return 255
    image = im.open(input_path)
    image = image.resize((IMG_DIM, IMG_DIM))
    image = image.convert("L") # grayscale
    image = image.point(lambda brightness: trinary_brightness(brightness))
    image.save(output_path)


################################################################################


def main():
    pass

if __name__ == "__main__":
    main()
