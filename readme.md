### ResNet18 Architecture

ResNet18 is a convolutional neural network that is 18 layers deep. It has 17 convolutional layers and 1 fully connected layer. It is trained on the ImageNet dataset. 

The architecture of ResNet18 is as follows:

1. Convolutional Layer 1: 64 filters of size 7x7 with stride 2
2. Max Pooling Layer 1: 3x3 filter with stride 2
3. Residual Block 1: 2 convolutional layers with 64 filters of size 3x3
4. Residual Block 2: 2 convolutional layers with 128 filters of size 3x3
5. Residual Block 3: 2 convolutional layers with 256 filters of size 3x3
6. Residual Block 4: 2 convolutional layers with 512 filters of size 3x3
7. Fully Connected Layer: 10 units (For CIFAR-10 dataset)

### References:
1. [Deep residual learning for image recognition (arxiv)](https://arxiv.org/abs/1512.03385)
2. [Torchvision Resnet](https://pytorch.org/hub/pytorch_vision_resnet/)

