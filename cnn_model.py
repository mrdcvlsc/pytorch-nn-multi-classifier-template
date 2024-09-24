import torch
from torch.backends import mps

################################################################
# If you run .device("cuda"), your tensor will be routed to
# the CUDA current device, which by default is the 0 device.

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if mps.is_available()
    else "cpu"
)

# You could try `torch.cuda.device_count()` to get the number of
# GPUs available and maybe `torch.cuda.get_device_name(device_id)`
# to get the name of the used device.

print(f"Using {DEVICE} device")

################################################################
# defining our model

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # All neural networks are implemented with nn.Module.
        # If the layers are sequentially used (self.layer3(self.layer2(self.layer1(x))),
        # you can leverage nn.Sequential to not have to define the forward function of the model.

        # In a more complicated module, you might need to use multiple sequential submodules.
        # For instance, take a CNN classifier, you could define a nn.Sequential for the CNN part,
        # then define another nn.Sequential for the fully connected classifier section of the model.

        # Input: Bx1x28x28

            # feature extraction block without downsampling
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = torch.nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = torch.nn.ReLU(inplace=True)
            # Bx8x28x28

            # feature processing block
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = torch.nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
            # Bx8x28x28

            # feature downsampling using max pooling
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            # Bx8x14x14

        self.flatten = torch.nn.Flatten()

        self.linear3 = torch.nn.Linear(8*14*14, 128)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.linear4 = torch.nn.Linear(128, 128)
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.linear5 = torch.nn.Linear(128, 10)

        self.softmax = torch.nn.Softmax(dim=1)

        # original from pytorch tutorial
        # self.linear_relu_stack = torch.nn.Sequential(
        #     torch.nn.Linear(28*28, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, 10),
        # )

    def forward(self, x):
        # print('x1.shape =', x.shape)

        x = self.relu1(self.bn1(self.conv1(x)))
        # print('x2.shape =', x.shape)

        x = self.relu2(self.bn2(self.conv2(x)))
        # print('x3.shape =', x.shape)

        x = self.maxpool(x)
        # print('x4.shape =', x.shape)

        x = self.flatten(x)
        # print('x5.shape =', x.shape)

        x = self.relu3(self.linear3(x))
        # print('x6.shape =', x.shape)

        x = self.relu4(self.linear4(x))
        # print('x6.shape =', x.shape)

        x = self.linear5(x)
        # print('x6.shape =', x.shape)

        x = self.softmax(x)

        return x

if __name__ == "__main__":

    model = NeuralNetwork().to(DEVICE)
    print(model)

    X: torch.Tensor = torch.rand(1, 1, 28, 28, device=DEVICE)
    y: torch.Tensor = torch.ones(1, dtype=torch.int64)
    logits: torch.Tensor = model(X)

    print('X shape =', X.shape)
    print('y shape =', y.shape)

    print('X dtype =', X.dtype)
    print('y dtype =', y.dtype)

    print('softmax output shape =', logits.shape)
    print('softmax output =', logits.clone().detach().numpy())

    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(logits, y)

    print('loss output shape =', loss.shape)
    print('loss output =', loss.clone().detach().numpy())

    pred_probab = torch.nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)

    print(f"Predicted class: {y_pred}")