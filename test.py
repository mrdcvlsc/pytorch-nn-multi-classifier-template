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
    print('model size: {:.3f}MB'.format(size_all_mb))

# print('loading vgg19_bn model')
# vgg19_bn = torchvision.models.vgg19_bn()
# print(vgg19_bn)
# print(f'vgg19_bn estimated size : {estimated_model_file_size(vgg19_bn)}')

# print('loading resnet152 model')
# resnet152 = torchvision.models.resnet152()
# # print(resnet152)
# print(f'resnet152 estimated size : {estimated_model_file_size(resnet152)}')

# print('loading inception_v3 model')
# inception_v3 = torchvision.models.inception_v3()
# # print(inception_v3)
# print(f'inception_v3 estimated size : {estimated_model_file_size(inception_v3)}')

print(torch.nn.MultiheadAttention(4, 2))