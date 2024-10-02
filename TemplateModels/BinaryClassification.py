import torch
from torch.backends import mps

class BinaryClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # All neural networks are implemented with nn.Module.
        # If the layers are sequentially used (self.layer3(self.layer2(self.layer1(x))),
        # you can leverage nn.Sequential to not have to define the forward function of the model.

        # In a more complicated module, you might need to use multiple sequential submodules.
        # For instance, take a CNN classifier, you could define a nn.Sequential for the CNN part,
        # then define another nn.Sequential for the fully connected classifier section of the model.

        # Input: Bx3x56x56

        self.conv_relu_stack1 = torch.nn.Sequential(
            # feature extraction block without downsampling
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            # Bx16x56x56

            # feature processing block
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            # Bx16x56x56

            # feature downsampling using max pooling
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # Bx16x28x28
        )

        self.conv_relu_stack2 = torch.nn.Sequential(
            # feature extraction block without downsampling
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            # Bx32x28x28

            # feature processing block
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            # Bx32x28x28

            # feature downsampling using max pooling
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # Bx32x14x14
        )

        self.conv_relu_stack3 = torch.nn.Sequential(
            # feature extraction block without downsampling
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            # Bx64x14x14

            # feature processing block 1
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            # Bx64x14x14

            # feature processing block 2
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            # Bx64x14x14

            # feature downsampling using max pooling
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # Bx64x7x7

            torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))
            # Bx64x7x7
        )

        self.flatten = torch.nn.Flatten()

        # 64x7x7 = 3_136
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(3_136, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):

        x = self.conv_relu_stack1(x)
        x = self.conv_relu_stack2(x)
        x = self.conv_relu_stack3(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)

        return x

if __name__ == "__main__":

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if mps.is_available()
        else "cpu"
    )

    model = BinaryClassification().to(DEVICE)
    print(model)
    print()

    batch_inputs: torch.Tensor = torch.rand(2, 3, 56, 56, device=DEVICE)
    batch_labels: torch.Tensor = torch.ones(2, dtype=torch.int64).float()
    batch_output: torch.Tensor = model(batch_inputs)
    batch_output.squeeze_()

    print(f'batch_inputs shape : {batch_inputs.shape} | dtype : {batch_inputs.dtype}')
    print(f'batch_labels shape : {batch_labels.shape} | dtype : {batch_labels.dtype}')
    print(f'batch_output shape : {batch_output.shape} | dtype : {batch_inputs.dtype}')
    print('batch_output =\n\n', batch_output.clone().detach().numpy(), '\n\n')
    print('batch_labels =\n\n', batch_labels.clone().detach().numpy(), '\n\n')

    loss_function = torch.nn.BCELoss()
    loss = loss_function(batch_output, batch_labels)

    print('loss output shape =', loss.shape)
    print('loss output =', loss.clone().detach().numpy())