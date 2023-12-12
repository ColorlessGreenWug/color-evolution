import numpy as np
from PIL import Image
from skimage import color

def hex_to_rgb(hex):
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

# read in the hex values
hex_values = []
with open("colors_selected.txt", "r") as f:
    for line in f:
        hex_str = line.strip()
        hex_str = hex_str.lstrip("#")
        hex_values.append(hex_str)

# convert to rgb
rgb_values = []
for h in hex_values:
    rgb_values.append(hex_to_rgb(h))

# convert to lab
lab_values = []
for rgb in rgb_values:
    lab_values.append(color.rgb2lab(*rgb))

# write out a txt file with the lab values
with open("colors_selected_lab.txt", "w") as f:
    for lab in lab_values:
        f.write(" ".join([str(x) for x in lab]) + "\n")