import torch
from torch.backends import mps

TEN_CLASS_VECTOR_LABEL = torch.Tensor([
    [0.4358898943540673, 0.0, 0.9],
    [-0.5265867068231264, -0.48239655906440165, 0.7],
    [0.07571289854915397, 0.8627094279033268, 0.4999999999999999],
    [0.5804136811532111, -0.7570468669310894, 0.3000000000000001],
    [-0.979777547038325, 0.1733088523981474, 0.1],
    [0.8395259183305308, 0.5340376695058868, -0.1000000000000001],
    [-0.24764672330212997, -0.9212334668463354, -0.3],
    [-0.3991571921844771, 0.7685528842749886, -0.5000000000000002],
    [0.6708095809104763, -0.244978583061278, -0.6999999999999998],
    [-0.4029128868115613, -0.16631658258025273, -0.9]]
)

class MCC3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input: Bx1x28x28

        self.conv_relu_stack = torch.nn.Sequential(
            # feature extraction block without downsampling
            torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.Tanh(),
            # Bx4x28x28

            # feature processing block
            torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.Tanh(),
            # Bx4x28x28

            torch.nn.AdaptiveAvgPool2d((14, 14))
            # Bx4x14x14
        )

        self.flatten = torch.nn.Flatten()

        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(4*14*14, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 3),
            torch.nn.Tanh()
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

    model = MCC3D().to(DEVICE)
    print(model)
    print()

    batch_inputs: torch.Tensor = torch.rand(3, 1, 28, 28, device=DEVICE)
    batch_labels: torch.Tensor = torch.Tensor([0, 2, 5]).long()
    batch_output: torch.Tensor = model(batch_inputs)

    print(f'batch_inputs shape : {batch_inputs.shape} | dtype : {batch_inputs.dtype}')
    print(f'batch_labels shape : {batch_labels.shape} | dtype : {batch_labels.dtype}')
    print(f'batch_output shape : {batch_output.shape} | dtype : {batch_inputs.dtype}')
    print('batch_output =\n\n', batch_output.clone().detach().numpy(), '\n\n')
    print('batch_labels =\n\n', batch_labels.clone().detach().numpy(), '\n\n')
    print('batch_label> =\n\n', TEN_CLASS_VECTOR_LABEL[batch_labels], '\n\n')

    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(batch_output, TEN_CLASS_VECTOR_LABEL[batch_labels])

    print('loss output shape =', loss.shape)
    print('loss output =', loss.clone().detach().numpy())

    pred_probab = torch.nn.Softmax(dim=1)(batch_output)
    y_pred = pred_probab.argmax(1)

    print(f"Predicted class: {y_pred}")