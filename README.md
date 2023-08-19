```python
import torch

# Import the torch library, which is a popular deep learning framework

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Check if a CUDA-enabled GPU is available using torch.cuda.is_available()
# If GPU is available, set device to 'cuda' to run code on GPU
# If GPU is not available, set device to 'cpu' to run code on CPU

# Print the selected device ('cuda' or 'cpu')
print(device)

# Import the necessary modules
# Import the datasets module from torchvision library
# The torchvision library provides popular datasets like MNIST, CIFAR10, etc.
# These datasets can be used for computer vision tasks
from torchvision import datasets

# Import the ToTensor class from the transforms module in torchvision
# The ToTensor transform is used to convert input data, such as images, into PyTorch tensors
# It ensures that the data is in the appropriate format for deep learning tasks
from torchvision.transforms import ToTensor
```