# read the rgb values of an image
from PIL import Image
import numpy as np

img = Image.open("colors.png")
rgb_img = img.convert('RGB')
img_array = np.array(rgb_img)

# get all unique colors that are bigger than a 8 x 8 square
unique_colors = []
for i in range(10, img_array.shape[0]-5, img_array.shape[0]//8+1):
    color_row = []
    i -= i // (img_array.shape[0]//4)
    for j in range(10, img_array.shape[1]-5, img_array.shape[1]//40): 
        j += j // (img_array.shape[1]//11)
        color_row.append(tuple(img_array[i, j]))
        img_array[i-3:i+3, j-3:j+3] = 0
    unique_colors.append(color_row)

unique_colors = np.array(unique_colors)

img = Image.fromarray(img_array)
img.save("colors2.png")

# create a new image with the unique colors by tiling a square with each
img_array = np.zeros((unique_colors.shape[0]*20, unique_colors.shape[1]*20, 3), dtype=np.uint8)
for i in range(unique_colors.shape[0]):
    for j in range(unique_colors.shape[1]):
        img_array[i*20:(i+1)*20, j*20:(j+1)*20] = unique_colors[i, j]

#save the image
img = Image.fromarray(img_array)
img.save("colors1.png")

# select the colors
selected_colors = np.zeros((3, 10, 3), dtype=np.uint8)
for t, i in enumerate([1, 4, 7]):
    for s, j in enumerate(range(0, 40, 4)):
        selected_colors[t, s] = unique_colors[i - s % 2, j + t]


# place the selected colors in an image
img_array = np.zeros((selected_colors.shape[0]*20, selected_colors.shape[1]*20, 3), dtype=np.uint8)
for i in range(selected_colors.shape[0]):
    for j in range(selected_colors.shape[1]):
        img_array[i*20:(i+1)*20, j*20:(j+1)*20] = selected_colors[i, j]

#save the image
img = Image.fromarray(img_array)
img.save("colors_selected.png")

# get the hex values of the colors and write to file
hex_values = []
for i in range(selected_colors.shape[0]):
    for j in range(selected_colors.shape[1]):
        hex_values.append('#%02x%02x%02x' % tuple(selected_colors[i, j]))
with open("colors_selected.txt", "w") as f:
    f.write("\n".join(hex_values))