import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import *
from loss import DiceLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
epochs = 200
LR = 1e-4


def train(model, model_path):
    # data preprocess
    train_transform = transforms.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        RandomFlip(),
        ToTensor(),
    ])
    val_transform = transforms.Compose([
        GrayscaleNormalization(mean=0.5, std=0.5),
        ToTensor(),
    ])

    # load data
    train_dataset = Dataset(imgs_dir='data_aug/train_img', labels_dir='data_aug/train_label', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataset = Dataset(imgs_dir='data_aug/val_img', labels_dir='data_aug/val_label', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print('Size of train set: ', len(train_dataset), 'Size of validation set: ', len(val_dataset))

    # get model
    model = model.to(device)

    # loss function
    criterion = nn.BCELoss().to(device)
    # criterion = DiceLoss().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=LR)

    for epoch in range(epochs):
        # train
        train_loss = 0
        avg_loss = 0
        model.train()

        for batch_num, data in tqdm(enumerate(train_loader)):
            imgs = data['img'].to(device)
            labels = data['label'].to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            avg_loss += loss.item()

            if (batch_num + 1) % 10 == 0:
                print('[%d, %d] loss: %.4f' % (epoch, batch_num + 1, avg_loss / 10))
                avg_loss = 0

        print('Epoch: %d Avg Training loss:%.4f' % (epoch, train_loss / len(train_loader)))

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_num, data in tqdm(enumerate(val_loader)):
                img = data['img'].to(device)
                label = data['label'].to(device)

                output = model(img)
                loss = criterion(output, label)
                val_loss += loss

        val_loss /= len(val_loader)
        print('Epoch: %d Avg Validation loss:%.4f' % (epoch, val_loss))

        torch.save(model.state_dict(), model_path)
