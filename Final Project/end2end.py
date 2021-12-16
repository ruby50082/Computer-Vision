import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model import VGG16, ResNet34

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model,dataloader,criterion,optimizer,epochs):
    model.train()
    for epoch in range(1,epochs+1):
        total_loss = 0
        correct = 0
        for images,targets in dataloader:
            images,targets = images.to(device),targets.to(device)
            predicts = model(images)
            loss = criterion(predicts, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(images)
            correct += predicts.max(dim=1)[1].eq(targets).sum().item()
        
        print('epoch {}, Loss: {:.4f}, Acc: {:.2f}'.format(epoch, total_loss/len(dataloader.dataset), correct/len(dataloader.dataset) * 100))


def test(model,dataloader):
    model.eval()
    correct=0
    for images,targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        predicts = model(images)

        correct += predicts.max(dim=1)[1].eq(targets).sum().item()

    print('Test Acc: {:.2f}'.format(correct/len(dataloader.dataset) * 100))


epochs_feature_extraction = 5
epochs_fine_tuning = 15
batch_size = 64
lr = 1e-3


if __name__=='__main__':

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_data = ImageFolder(os.path.join('hw5_data', 'train'), transform=train_transforms)
    test_data = ImageFolder(os.path.join('hw5_data', 'test'), transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    print('Training:')
    # model = VGG16(num_class=15)
    model = ResNet34(num_class=15)
    model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # feature extraction
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9)
    train(model, train_loader, criterion, optimizer,epochs_feature_extraction)

    print('Fine Tuning:')
    # finetuning
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    train(model, train_loader, criterion, optimizer, epochs_fine_tuning)

    # save model weight
    model_wts = copy.deepcopy(model.state_dict())
    torch.save(model_wts, 'model.pt')

    # test
    with torch.no_grad():
        test(model, test_loader)