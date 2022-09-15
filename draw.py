from PIL import Image, ImageDraw

# https://note.nkmk.me/en/python-pillow-imagedraw/


im_h = 64
im_w = 64
img_word_len = 8
scale = pix_d = 10

im_h *= scale
im_w *= scale
img_word_len *= scale


im = Image.new('RGB', (im_w, im_h), (128, 128, 128))
print(im.size)
draw = ImageDraw.Draw(im)

# x, y = 0, 0 # top corner 
# pix_d = scale
# draw.rectangle([x, y, x+pix_d, y+pix_d], fill=(0, 0, 0), outline=(0, 255, 0))
# draw.rectangle([100,100,200,200], outline=(0, 255, 0))
# im.save('test.png')

x, y = 0, 0
for i in range(im_w // img_word_len):

    rgb = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)

    left_offset = i * img_word_len 
    draw.rectangle([x+left_offset, y, x+left_offset+img_word_len-1, y+pix_d-1],
                    outline=rgb)

x, y = 0, pix_d
for i in range(im_w // img_word_len):

    rgb = (0, 255, 0) if i % 2 == 1 else (255, 0, 0)

    left_offset = i * img_word_len 
    draw.rectangle([x+left_offset, y, x+left_offset+img_word_len-1, y+pix_d-1],
                    outline=rgb)

im.save('test.png')
