#%%
import torch
from torch import nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import sys

#hyperparameters
batch_size = 64
learning_rate = 1e-3
epochs = 5

# prepare data
train_datapath = "/home/zht/Datasets/proprioception/trainSet_c7_corrected.hdf5"
valid_datapath = "/home/zht/Datasets/proprioception/validSet_c7_corrected.hdf5"
test_datapath = "/home/zht/Datasets/proprioception/testSet_c7_ver_2.hdf5"
def load_data(datapath):
    f = h5py.File(datapath, 'r')
    timeStamps = f['timeStamps']['timeStamps'][:]
    signals = f['signals']['signals'][:]
    labels = f['labels']['labels'][:]
    images = f['images']['images'][:]
    print(timeStamps.shape, "# of timeStamps")
    print(signals.shape,"# of signals")
    print(images.shape,"# of images")
    print(labels.shape,"# of labels")
    f.close()
    images = images.reshape(images.shape[0], 224, 224, 3)/255.0 # normalize to 0-1
    signals = signals.reshape(signals.shape[0], 100, 9)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # normilize helps a little
    ])
    data = ImageDataset(labels, images, transform=transform)
    print(len(data),"# data length")
    return data
print("trainset:")
training_data = load_data(train_datapath)
print("validset:")
valid_data = load_data(valid_datapath)
print("testset:")
test_data = load_data(test_datapath)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
labelp = ['asphalt','grass','gravel','pavement','sand','brick','coated floor']
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}") 
# img = train_features[0]
# label = train_labels[0]
# plt.imshow(img.permute(1,2,0))
# print(labelp[label])

# build model
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()
print("classifier changes: ", model.classifier)
# for child in model.named_children():
#     print(child)
classifier = nn.Sequential(
    nn.Dropout(0.25),
    nn.Linear(1280, 32),
    nn.Linear(32, 7),
)
model.classifier = classifier
# print(model(train_features).shape)
# only finetune classifier
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# train and validate
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
for epoch in range(epochs):
    loss_list = []
    total_correct = 0
    total = 0
    pbar = tqdm(total=len(train_dataloader))
    for batch, (X, y) in enumerate(train_dataloader):
        imgs = X.to(device)
        y_hat = model(imgs)
        y = y.type(torch.LongTensor).to(device)
        # print(y_hat.shape)
        # print(type(y))
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
        total_correct += correct
        total += len(y)
        loss_list.append(loss.item())
        pbar.set_description(f"epoch {epoch+1}")
        pbar.set_postfix(loss=f"{loss.item():>7f}", accuracy=f"{total_correct/total:>7f}")
        pbar.update(1)
    pbar.close()
    print(f"In epoch {epoch+1}, total train accuracy: {total_correct/total:>7f}, mean loss: {np.mean(loss_list):>7f}")

    pbar = tqdm(total = len(valid_dataloader))
    loss_list = []
    total_correct = 0
    total = 0
    for batch, (X, y) in enumerate(valid_dataloader):
        imgs = X.to(device)
        y_hat = model(imgs)
        y = y.type(torch.LongTensor).to(device)
        loss = loss_fn(y_hat, y)
        correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
        total_correct += correct
        total += len(y)
        loss_list.append(loss.item())
        pbar.set_description(f"epoch {epoch+1}")
        pbar.set_postfix(accuracy=f"{total_correct/total:>7f}")
        pbar.update(1)
    pbar.close()
    print(f"In epoch {epoch+1}, total valid accuracy: {total_correct/total:>7f}, mean loss: {np.mean(loss_list):>7f}")

# test
pbar = tqdm(total = len(test_dataloader))
loss_list = []
total_correct = 0
total = 0
for batch, (X, y) in enumerate(test_dataloader):
    imgs = X.to(device)
    y_hat = model(imgs)
    y = y.type(torch.LongTensor).to(device)
    loss = loss_fn(y_hat, y)
    correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
    total_correct += correct
    total += len(y)
    loss_list.append(loss.item())
    pbar.set_description(f"epoch {epoch+1}")
    pbar.set_postfix(accuracy=f"{total_correct/total:>7f}")
    pbar.update(1)
pbar.close()
print(f"total test accuracy: {total_correct/total:>7f}, mean loss: {np.mean(loss_list):>7f}")
#%%
# save model
torch.save(model.state_dict(), "./model/model1.pth")

# %%
