import torch
import torch.nn.functional as F

# edge detection - 1D convolution
def detect_edges(image, filter_weights): 
    results = []
    for i in range(len(image)-1):
        window = [image[i], image[i+1]]
        result = window[0]*filter_weights[0] + window[1]*filter_weights[1]
        results.append(result)
    return results

# horizontal edge detection - 2D convolution
def detect_horizontal(image2d, filter2d, stride=1):
    num_rows = len(image2d)
    num_cols = len(image2d[0])
    results = []

    # detecting horizontal edges by sliding filter vertically (down the columns)
    for col in range(num_cols):
        for row in range(0, num_rows-1, stride):
            window = [image2d[row][col], image2d[row+1][col]]
            result = window[0]*filter2d[0] + window[1]*filter2d[1]
            results.append(result)

    return results

# vertical edge detection - 2D convolution
def detect_vertical_edges(image2d, filter2d, stride=1):
    num_rows = len(image2d)
    num_cols = len(image2d[0])
    results = []

    # detecting vertical edges by sliding filter horizontally (across the rows)
    for row in range(num_rows):  
        for col in range(0, num_cols-1, stride):  
            window = [image2d[row][col], image2d[row][col+1]]  
            result = window[0]*filter2d[0] + window[1]*filter2d[1]
            results.append(result)

    return results


# padding - border handling -> adds '0' padding around the image
def add_padding(image2d, pad_size=1):
    num_rows = len(image2d)
    num_cols = len(image2d[0])
    new_rows = num_rows + 2*pad_size
    new_cols = num_cols + 2*pad_size

    # creating a padded image filled with zeros
    padded = [[0 for _ in range(new_cols)] for _ in range(new_rows)]
    
    # image pasted into the center
    for i in range(num_rows):
        for j in range(num_cols):
            padded[i + pad_size][j + pad_size] = image2d[i][j]
    
    return padded

# multi channel convolution - for RGB images
def detect_edges_rgb(image_rgb, filter_rgb):

    red_res = detect_vertical_edges(image_rgb[0], filter_rgb[0])
    green_res = detect_vertical_edges(image_rgb[1], filter_rgb[1])
    blue_res = detect_vertical_edges(image_rgb[2], filter_rgb[2])

    combined = []

    for i in range(len(red_res)):
        combined.append(red_res[i]+green_res[i]+blue_res[i])

    return combined

red_channel = [[0,1,0,1], [0,1,0,1], [0,1,0,1], [0,1,0,1]]
green_channel = [[1,0,1,0], [1,0,1,0], [1,0,1,0], [1,0,1,0]]
blue_channel = [[0,1,0,1], [0,1,0,1], [0,1,0,1], [0,1,0,1]]

image_rgb = [red_channel, green_channel, blue_channel]
filter_rgb = [[1,-1], [1,-1], [1,-1]]

# CNN layer -> convolution + activation
def cnn_layer(image, filters, stride=1, activation='relu'):
    conv_op = detect_vertical_edges(image, filters, stride)
    
    activated = []
    # to remove all the negative values
    if activation=='relu':
        for i in range(len(conv_op)):
            activated.append(max(0, conv_op[i]))
    return activated

# applying multiple filters to the same image (in parallel)
def cnn_layer_multi(image, filter_list, stride=1, activation='relu'):
    outputs = []

    for f in filter_list:
        res = cnn_layer(image, f, stride, activation)
        outputs.append(res)

    return outputs

# stacking the layers -> reshaping 1D convolution output to 2D for next layer
def reshape_to_2d(flat_list, num_cols):
    reshaped = []
        
    for i in range(0, len(flat_list), num_cols):
        chunk = flat_list[i:i+num_cols]
        reshaped.append(chunk)

    return reshaped

# 2 layer CNN -> hierarchial feature
def simple_cnn(image, layer1_filters, layer2_filters, stride=1):
    # applying 1st set of filters
    layer1_output = cnn_layer_multi(image, layer1_filters, stride, activation='relu')
    layer1_flat = layer1_output[0]

    # reshape for next layer
    num_cols = len(image[0]) - 1
    layer1_2d = reshape_to_2d(layer1_flat, num_cols)

    # applying 2nd set of filters to layer 1 features
    layer2_output = cnn_layer_multi(layer1_2d, layer2_filters, stride, activation='relu')
    
    return layer2_output

# layer pooling -> to reduce dimensions by 50%
def max_pool_2x2(image):
    pooled = []
    num_rows = len(image)
    num_cols = len(image[0])

    # sliding using 2x2 window using stride = 2 (non overlapping)
    for row in range(0, num_rows, 2):
        for col in range(0, num_cols, 2):
            window = [image[row][col], image[row][col+1], image[row+1][col], image[row+1][col+1]]
            pooled.append(max(window))

    return pooled

# complete CNN pipeline
def complete_cnn(image, conv_filters, pool=True):
    # convolution + activation
    conv_output = cnn_layer_multi(image, conv_filters, stride=1, activation='relu')

    conv_flat = conv_output[0]
    num_cols = len(image[0])-1
    conv_2d = reshape_to_2d(conv_flat, num_cols)

    # optional pooling -> only if the dimensions are the even (ie 4x4 etc)
    if pool and len(conv_2d) % 2 == 0 and len(conv_2d[0]) % 2 == 0:
        pooled = max_pool_2x2(conv_2d)
        return pooled
    
    return conv_2d

test_img = [
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1]
    ]
filters = [[1, -1]]
filter_vertical = [1, -1]

result = complete_cnn(test_img, filters, pool=True)
print(f"complete cnn output : {result}")

test_img_tensor = torch.tensor(test_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
filter_tensor = torch.tensor([[filter_vertical]], dtype=torch.float32).unsqueeze(0)
pytorch_result = F.conv2d(test_img_tensor, filter_tensor)

print(f"pytorch output : {torch.relu(pytorch_result)}")