print('loading libraries')
import torch
import torchvision

def estimated_model_file_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MiB'.format(size_all_mb))

import inspect

print('loading model')
selected_model = torchvision.models.resnet18()
print(f'\nModel: {selected_model.__class__.__name__} \n')
print(selected_model)
print('\nCode Method:\n')
print(inspect.getsource(selected_model.__class__))