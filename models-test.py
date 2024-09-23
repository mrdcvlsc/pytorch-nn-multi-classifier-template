print('loading libraries')
import torchvision

print('loading vgg19_bn model')
vgg19_bn = torchvision.models.vgg19_bn()
print(vgg19_bn)
print()

print('loading resnet152 model')
resnet152 = torchvision.models.resnet152()
print(resnet152)
print()