from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from simple_cnn import *
import math

# load image
img = Image.open('./test.png').convert('L')
width, height = img.size

# resize it to nearest value of 2
def nearest_power_of_2(n):
    return 2**round(math.log(n))

new_width = nearest_power_of_2(width)
new_height = nearest_power_of_2(height)
img = img.resize((new_width, new_height))

# converting PIL to list
img_array = np.array(img)
img_list = img_array.tolist()

# applying vertical edge detection
filter_vertical = [1, -1]
vertical_edges = detect_vertical_edges(img_list, filter_vertical)
# reshaping 1D output to 2D
v_edges_2d = reshape_to_2d(vertical_edges, num_cols=new_height-1)

# applying horizontal edge detection
filter_horizontal = [[1], [-1]]
horizontal_edges = detect_horizontal_edges(img_list, filter_horizontal)

# reshaping 1D output to 2D
h_edges_2d = reshape_to_2d(horizontal_edges, num_cols=new_width)

# plotting all the image for comparison
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# original image
axes[0].imshow(img_array, cmap='gray')
axes[0].set_title('original grayscale img')

# image with vertical edges detected
vert_edges_array = np.array(v_edges_2d) 
axes[1].imshow(vert_edges_array, cmap='gray')
axes[1].set_title('vertical edges')

# image with horizontal edges detected
hori_edges_array = np.array(h_edges_2d)
axes[2].imshow(hori_edges_array, cmap='gray')
axes[2].set_title('horizontal edges')

# combined edge detection
v_small = vert_edges_array[:new_width-1, :new_height-1]
h_small = hori_edges_array[:new_width-1, :new_height-1]
mag = np.sqrt(v_small**2+h_small**2)

axes[3].imshow(mag, cmap='hot')
axes[3].set_title('combined edge magnitude')

plt.tight_layout()
plt.show()