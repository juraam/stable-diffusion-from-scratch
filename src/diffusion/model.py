import torch
import torch.nn

class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3
        )
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3
        )
        self.relu_2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu_1(self.conv_1(x))
        return self.relu_2(self.conv_2(x))

class UNet(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.resnet_left_1 = ResnetBlock(in_channels, out_channels=3)
        self.down_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.resnet_left_2 = ResnetBlock(in_channels=3, out_channels=3)
        self.down_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.resnet_left_3 = ResnetBlock(in_channels=3, out_channels=3)
        self.down_3 = torch.nn.MaxPool2d(kernel_size=2)

        self.backbone = ResnetBlock(in_channels=3, out_channels=3)

        self.up_1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=2)
        self.resnet_right_1 = ResnetBlock(in_channels, out_channels=3)
        self.up_2 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=2)
        self.resnet_right_2 = ResnetBlock(in_channels, out_channels=3)
        self.up_3 = torch.nn.MaxPool2d(kernel_size=2)
        self.resnet_right_3 = ResnetBlock(in_channels, out_channels=3)

    def forward(self, x):
        x_1 = self.resnet_left_1(x)
        x = self.down_1(x_1)
        x_2 = self.resnet_left_2(x)
        x = self.down_2(x_2)
        x_3 = self.resnet_left_3(x)

        x = self.down_3(x_3)

        x = self.backbone(x)
        x = self.up_1(x)
        x = self.resnet_right_1(x + x_3)
        x = self.up_2(x)
        x = self.resnet_right_2(x + x_2)
        x = self.up_3(x)
        x = self.resnet_right_1(x + x_1)

        return x
