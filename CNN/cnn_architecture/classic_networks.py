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

if __name__ == "__main__":
    model = LeNet5()
    # 1 input channel, 1 output channel, dims = 32x32
    test_input = torch.randn(1, 1, 32, 32)
    output = model(test_input)

    print(f"input shape : {test_input.shape}")
    print(f"output shape : {output.shape}")
    print("model created successfully")