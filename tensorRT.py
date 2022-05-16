import torch, tensorrt
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet
from time import time
# create some regular pytorch model...
model = alexnet(pretrained=False).eval().cuda()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()
torch.Tensor
# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])

t = time()
y = model(x)#.detach()
tt = time()
print(y.argmax(), tt-t)

s = time()
y_trt = model_trt(x)#.detach()
e = time()
print(y_trt.argmax(), e-s)


