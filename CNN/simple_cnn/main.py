from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from simple_cnn import *

img = Image.open('./test.png').convert('L')
img = img.resize((64, 64))

img_array = np.array(img)
img_list = img_array.tolist()

filter_vertical = [1, -1]
vertical_edges = detect_vertical_edges(img_list, filter_vertical)

filter_horizontal = [[1], [-1]]
horizontal_edges = detect_horizontal_edges(img_list, filter_horizontal)

# edges_2d = reshape_to_2d(vertical_edges, num_cols=63)
edges_2d = reshape_to_2d(horizontal_edges, num_cols=64)
# for row in edges_2d:
#     print(row)

# vert_edges_array = np.array(edges_2d)
hori_edges_array = np.array(edges_2d)

# plt.imshow(img_array, cmap='gray')
# plt.title('original grayscale img')

# plt.imshow(vert_edges_array, cmap='gray')
# plt.title('vertical edges')
plt.imshow(hori_edges_array, cmap='gray')
plt.title('horizontal edges')
plt.show()