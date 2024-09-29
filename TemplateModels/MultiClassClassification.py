import torch
from torch.backends import mps

class MultiClassClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # All neural networks are implemented with nn.Module.
        # If the layers are sequentially used (self.layer3(self.layer2(self.layer1(x))),
        # you can leverage nn.Sequential to not have to define the forward function of the model.

        # In a more complicated module, you might need to use multiple sequential submodules.
        # For instance, take a CNN classifier, you could define a nn.Sequential for the CNN part,
        # then define another nn.Sequential for the fully connected classifier section of the model.

        # Input: Bx1x28x28

        self.conv_relu_stack = torch.nn.Sequential(
            # feature extraction block without downsampling
            torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            # Bx4x28x28

            # feature processing block
            torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            # Bx4x28x28

            # feature downsampling using max pooling
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # Bx4x14x14
        )

        self.flatten = torch.nn.Flatten()

        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(4*14*14, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 10),
            torch.nn.Softmax(dim=1),
        )


    def forward(self, x):

        x = self.conv_relu_stack(x)
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

    model = MultiClassClassification().to(DEVICE)
    print(model)
    print()

    batch_inputs: torch.Tensor = torch.rand(3, 1, 28, 28, device=DEVICE)
    batch_labels: torch.Tensor = torch.ones(3, dtype=torch.int64)
    batch_output: torch.Tensor = model(batch_inputs)

    print(f'batch_inputs shape : {batch_inputs.shape} | dtype : {batch_inputs.dtype}')
    print(f'batch_labels shape : {batch_labels.shape} | dtype : {batch_labels.dtype}')
    print(f'batch_output shape : {batch_output.shape} | dtype : {batch_inputs.dtype}')
    print('batch_output =\n\n', batch_output.clone().detach().numpy(), '\n\n')

    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(batch_output, batch_labels)

    print('loss output shape =', loss.shape)
    print('loss output =', loss.clone().detach().numpy())

    pred_probab = torch.nn.Softmax(dim=1)(batch_output)
    y_pred = pred_probab.argmax(1)

    print(f"Predicted class: {y_pred}")