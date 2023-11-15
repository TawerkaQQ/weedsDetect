import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from utils import set_seed, load_model, save, get_model, update_optimizer, get_data
from epoch import train_epoch, val_epoch, test_epoch
from cli import add_all_parsers

# list for create graphics
arrval = []
lossval = []
arrtrain = []
losstrain = []
arrepoch = []

BATCH_SIZE = 64
learning_rate = 0.000009
image_size = (224, 224)
epoch = 400

dataset_train_path = "/home/cyber/Downloads/PlantNet-300K/dataset_ambrosia/train"
dataset_val_path = "/home/cyber/Downloads/PlantNet-300K/dataset_ambrosia/val"
dataset_test_path = "/home/cyber/Downloads/PlantNet-300K/dataset_ambrosia/test"

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])


dataset_train = ImageFolder(dataset_train_path, transform=transform)
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

dataset_val = ImageFolder(dataset_val_path, transform=transform)
val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)

dataset_test = ImageFolder(dataset_test_path, transform=transform)
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)




