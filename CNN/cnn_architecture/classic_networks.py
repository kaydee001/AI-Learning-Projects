import torch
import torch.nn as nn

# calling base class nn.Module such that LeNet5 inherits from the module
class LeNet5(nn.Module):
    def __init__(self):
        # calling nn.Module first 
        super(LeNet5, self).__init__()

        # convolution layers
        # input channels, output channels, kernel size : which is both width and height (ex : 5x5)
        # 1 input, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6 input, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)

        # pooling layer -> 2x2
        self.pool = nn.AvgPool2d(2)

        # fully connected layers
        # forced architecture -> LeNet design choice; the output of last layer must match input of next layer
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # activation
        # to learn more complex patterns : keeps the curve bending, instead of linear (straight)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input 32x32x1
        x = self.conv1(x)
        # 32-5+1 = 28
        x = self.relu(x)
        # 28x28x6
        x = self.pool(x)
        # 28/2=14x14x6
        
        # 14x14x6
        x = self.conv2(x)
        # 14-5+1=10
        x = self.relu(x)
        #=10x10x16
        x = self.pool(x)
        #=10/2=5x5x16

        # flatting -> 5x5x16=400
        x = torch.flatten(x, start_dim=1)

        # fc layers block
        x = self.fc1(x) #400 -> 120
        x = self.relu(x)
        x = self.fc2(x) # 120 -> 84
        x = self.relu(x)
        x = self.fc3(x) # 84 -> 10

        # 10 values
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        # RGB image -> 227x227x3 channels, 96 filters, 11x11 kernel, step/stride=4, no padding
        # most of this is by design choice like the 96 filters
        # bigger stride -> faster downsampling
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        # output_size = (input_size-kernel_size)/stride + 1
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        # preserving the dimensions in conv3 and conv4 
        # kernel_size = k -> padding=(k-1)/2
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # compressing the spatial features -> 9216=6x6x256 flattened conv features
        self.fc1 = nn.Linear(9216, 4096)
        # learning complex combinations
        self.fc2 = nn.Linear(4096, 4096)
        # final classification
        self.fc3 = nn.Linear(4096, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # conv -> relu -> pool
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # feature learning 1
        x = self.conv3(x)
        x = self.relu(x)

        # feature learning 2
        x = self.conv4(x)
        x = self.relu(x)

        # feature learning 3 + downsampling using pool
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)

        # flattens channels, height and width
        x = torch.flatten(x, start_dim=1)
        
        # linear -> relu -> dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # output layer -> relu will replace the negative values to 0
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model1 = LeNet5()
    # 1 input channel, 1 output channel, dims = 32x32
    test_input1 = torch.randn(1, 1, 32, 32)
    output1 = model1(test_input1)

    print(f"input shape : {test_input1.shape}")
    print(f"output shape : {output1.shape}")
    print("lenet5 model created successfully")
    
    model2 = AlexNet()
    # batch size, channels, height, width
    test_input2 = torch.randn(1, 3, 227, 227)
    output2 = model2(test_input2)

    print(f"input shape : {test_input2.shape}")
    print(f"output shape : {output2.shape}")
    print("alex net model created successfully")
