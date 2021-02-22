import torch
import torchvision
import torchvision.transforms as transforms
import math
from transformations import Compose, ToTensor, RandomHorizontalFlip, EqualizeAdaptiveHistogramEqualization, Resize
from TeslaSiemensDataset import TeslaSiemensDataset
from model import SegNet
from trainloop import train
from  DiceLoss import dice_loss
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn


def main():
    transform = Compose([
        EqualizeAdaptiveHistogramEqualization(),
        ToTensor(),
        Resize((50, 50)),
        RandomHorizontalFlip()
    ])

    BATCH_SIZE = 5

    trainset = TeslaSiemensDataset(root_dir='../data/siemens_reduced/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=False, num_workers=2)

    testset = TeslaSiemensDataset(root_dir='../data/siemens_reduced/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,shuffle=False, num_workers=2)

    net = SegNet(1,4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    criterion = dice_loss
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(model=net, optimizer=optimizer, loss_fn=criterion, train_loader=trainloader, val_loader=testloader, epochs=3, device=device)



if __name__ == "__main__":
    main()