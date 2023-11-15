from PIL import Image
import torch
from torchvision import transforms
import os
import numpy as np
import PIL
from torchvision import datasets
from torch.utils.data import Dataset 
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

def test_models(latest_model, best_model, test_loader):

    latest_model.load_state_dict(torch.load(latest_model['model_state_dict']))
    latest_model.eval()

    best_model.load_state_dict(torch.load(best_model['model_state_dict']))
    best_model.eval()
    
    acc_latest_model = 0
    acc_best_model = 0
    all_photo = len(test_loader.dataset)

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs_latest_model = latest_model(inputs)
            _, predicted_latest_model = torch.max(outputs_latest_model, 1)
            acc_latest_model += (predicted_latest_model == labels).sum().item()

            outputs_best_model = best_model(inputs)
            _, predicted_best_model = torch.max(outputs_best_model, 1)
            acc_best_model += (predicted_best_model == labels).sum().item()

    acc_latest_model = acc_latest_model/all_photo
    acc_best_model = acc_best_model/all_photo

    
    print(f'Accuracy for the latest model: {acc_latest_model:.4f}')
    print(f'Accuracy for the best model: {acc_best_model:.4f}')


train_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        np.asarray,
        np.copy,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])

test_dataset = datasets.ImageFolder("/home/cyber/Downloads/PlantNet-300K/testForREsNet18/")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=168, shuffle=False, num_workers=0)
test_loader.dataset.transform = train_transforms

latest_model = resnet18()
best_model = resnet18()

latest_model = torch.load('/home/cyber/Desktop/project_AmbrosiaSystem/checkpoints/latest_model.pt')
best_model = torch.load('/home/cyber/Desktop/project_AmbrosiaSystem/checkpoints/best_model.pt')

test_models(latest_model, best_model, test_loader)