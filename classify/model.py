""" Contains the Convolutional NN that we are using to run the MNIST
image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()

        # The input layer for our CNN. It takes in 1 image as input
        # and output 32 convolutional features with a kernel of size 3.
        self.conv1 = nn.Conv2d(1, 32, 3, 1)

        # The 2nd conv layer will output 64 convolutional features with
        # kernel 3.
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Add regularization to the model to ensure that we do not over fit
        # on the specifics of each classes different images.
        self.dropout1 = nn.Dropout2d(0.25)

        # Regular linear connected layers that will go between each convolutional
        # layer.
        self.linear1 = nn.Linear(9216, 128)

        # This is the output layer of our network. It will output 2, since we are
        # predicting the output for 2 classes as a percentage measure.
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # Apply 2-dim max pooling to get the maximun relevant feature value.
        x = F.max_pool2d(x, 2)

        # Apply dropout regularization after a full run through the first layer.
        x = self.dropout1(x)

        # Flatten the data into a 1 dimensional array so that we can feed it forward
        # to our hidden linear layer.
        x = torch.flatten(x, 1)

        # Run through linear layers and apply regularization and activation RELU.
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)

        # Softmax activation for outputting a probability between 0 - 1 for each
        # possible class.
        return F.log_softmax(x, dim=1)
