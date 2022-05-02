# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

# Super Resolution model definition in PyTorch
import torch.nn as nn
import torch.nn.init as init


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1))
        self.conv2 = nn.Conv2d(64, 1000, (3, 3), (1, 1))
        # self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        # self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        # self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # self._initialize_weights()
        # 12345

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = torch.reshape(x, (-1, 1000))
        # x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        # x = self.pixel_shuffle(self.conv4(x))
        return x

    # def _initialize_weights(self):
    #     init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.conv4.weight)


# Create the super-resolution model by using the above model definition.
torch_model = SuperResolutionNet(upscale_factor=3)

batch_size = 1

# Input to the model
x = torch.randn(batch_size, 3, 5, 5, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(
    torch_model,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=12,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["gpu_0/data_0"],  # the model's input names
    output_names=["gpu_0/softmax_1"],  # the model's output names
    dynamic_axes={
        "gpu_0/data_0": {0: "batch_size"},  # variable length axes
        "gpu_0/softmax_1": {0: "batch_size"},
    },
)
