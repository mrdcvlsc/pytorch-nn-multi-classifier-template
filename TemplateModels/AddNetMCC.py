import torch
from torch.backends import mps

import torch
import torch.nn as nn
import math

class LinearAdderV1(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearAdderV1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Define the learnable weight parameter (matrix)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Define the learnable bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Use a similar initialization as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Perform the "dot sum" operation
        # Shape of input: (batch_size, in_features)
        # Shape of weight: (out_features, in_features)
        # We need to perform element-wise addition for each output feature
        # and then sum across the input features

        # Add input with weight in an element-wise manner, then sum across the last dimension
        output = torch.sum(input.unsqueeze(1) + self.weight, dim=-1)

        # Add bias if it exists
        if self.bias is not None:
            output *= self.bias

        return output

class LinearAdderV2(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearAdderV2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameter (for element-wise addition, same size as standard Linear layer)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias (optional, similar to nn.Linear)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights uniformly and bias, same as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Input shape: [batch_size, in_features]
        # Weight shape: [out_features, in_features]
        
        # Instead of dot product (matrix multiplication), perform element-wise addition
        # and then sum over the 'in_features' dimension
        output = torch.sum(input.unsqueeze(1) + self.weight, dim=2)
        
        # Add bias if required
        if self.bias is not None:
            output *= self.bias
        
        return output

# Example usage
input_tensor = torch.tensor([
    [  1.0, -6.0,  3.0, -4.0],
    [ -4.0,  2.0, -2.0,  4.0],
    [  7.0, -3.0,  5.0, -1.0]
])  # Batch of 3 samples, each with 4 input features

linear_add_layer_v1 = LinearAdderV1(4, 2, bias=False)  # Custom linear layer with 4 input features and 2 output features
linear_add_layer_v2 = LinearAdderV2(4, 2, bias=False)  # Custom linear layer with 4 input features and 2 output features
linear_pt = torch.nn.Linear(4, 2, bias=False)

with torch.no_grad():
    linear_add_layer_v1.weight.copy_(torch.tensor([
        [-1.0,  5.0,  1.0,  2.0],
        [ 3.0, -2.0, -3.0, -4.0]
    ]))

    linear_add_layer_v2.weight.copy_(torch.tensor([
        [-1.0,  5.0,  1.0,  2.0],
        [ 3.0, -2.0, -3.0, -4.0]
    ]))

    linear_pt.weight.copy_(torch.tensor([
        [-1.0,  5.0,  1.0,  2.0],
        [ 3.0, -2.0, -3.0, -4.0]
    ]))

print(f'input {input_tensor.shape} : \n', input_tensor, '\n')

print('linear_add_layer_v1.weight : \n', linear_add_layer_v1.weight, '\n')
print('linear_add_layer_v2.weight : \n', linear_add_layer_v2.weight, '\n')
print('linear_pt.weight           : \n', linear_pt.weight, '\n')

output_v1 = linear_add_layer_v1(input_tensor)
output_v2 = linear_add_layer_v2(input_tensor)
linpytout = linear_pt(input_tensor)

print('output_v1.size() = ', output_v1.size())
print('output_v2.size() = ', output_v2.size())
print('linpytout.size() = ', linpytout.size(), '\n')

print('output_v1 :\n', output_v1, '\n')
print('output_v2 :\n', output_v2, '\n')
print('linpytout :\n', linpytout, '\n')

class AddNetMCC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input: Bx1x28x28 = 784

        self.flatten = torch.nn.Flatten()

        self.linear_relu_stack = torch.nn.Sequential(
            LinearAdderV1(784, 784),
            torch.nn.ReLU(inplace=True),
            LinearAdderV1(784, 10),
            torch.nn.ReLU(inplace=True),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x

class NetMCC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input: Bx1x28x28 = 784

        self.flatten = torch.nn.Flatten()

        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(784, 784),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(784, 10),
            torch.nn.ReLU(inplace=True),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
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

    model = AddNetMCC().to(DEVICE)
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