# import dependencies
import tensorflow as tf
import torch

# no suport for AMD GPUs on OSX with ROCm
device = torch.device("cpu")