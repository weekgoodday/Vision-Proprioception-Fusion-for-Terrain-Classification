#%%
import torch 
from torch import nn
import h5py
from torchvision import transforms
from dataset import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np 

#hyperparameters
batch_size = 64

# model architecture
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')
# for child in model.named_children():
#     print(child)
classifier = nn.Sequential(
    nn.Dropout(0.25),
    nn.Linear(1280, 32),
    nn.Linear(32, 7),
)
model.classifier = classifier
print("classifier changes: ", model.classifier)
# print(model(train_features).shape)
# only finetune classifier
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False
model.load_state_dict(torch.load("./model/model999.pth"))

# prepare data
test_datapath1 = "/home/zht/Datasets/proprioception/darkSet_c7_ver_2.hdf5"
test_datapath2 = "/home/zht/Datasets/proprioception/sim_fog_ver_2.hdf5"
test_datapath3 = "/home/zht/Datasets/proprioception/sim_sun_ver_2.hdf5"
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
print("darkset:")
test_data1 = load_data(test_datapath1)
print("fogset:")
test_data2 = load_data(test_datapath2)
print("sunset:")
test_data3 = load_data(test_datapath3)
dark_dataloader = DataLoader(test_data1, batch_size=batch_size, shuffle=False)
fog_dataloader = DataLoader(test_data2, batch_size=batch_size, shuffle=False)
sun_dataloader = DataLoader(test_data3, batch_size=batch_size, shuffle=False)

# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# test
loss_fn = nn.CrossEntropyLoss()
model.eval()
pbar = tqdm(total = len(dark_dataloader))
loss_list = []
total_correct = 0
total = 0
for batch, (X, y) in enumerate(dark_dataloader):
    imgs = X.to(device)
    y_hat = model(imgs)
    y = y.type(torch.LongTensor).to(device)
    loss = loss_fn(y_hat, y)
    correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
    total_correct += correct
    total += len(y)
    loss_list.append(loss.item())
    pbar.set_postfix(accuracy=f"{total_correct/total:>7f}")
    pbar.update(1)
pbar.close()
print(f"total test accuracy: {total_correct/total:>7f}, mean loss: {np.mean(loss_list):>7f}")
# %%
pbar = tqdm(total = len(fog_dataloader))
loss_list = []
total_correct = 0
total = 0
for batch, (X, y) in enumerate(fog_dataloader):
    imgs = X.to(device)
    y_hat = model(imgs)
    y = y.type(torch.LongTensor).to(device)
    loss = loss_fn(y_hat, y)
    correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
    total_correct += correct
    total += len(y)
    loss_list.append(loss.item())
    pbar.set_postfix(accuracy=f"{total_correct/total:>7f}")
    pbar.update(1)
pbar.close()
print(f"total test accuracy: {total_correct/total:>7f}, mean loss: {np.mean(loss_list):>7f}")
# %%
pbar = tqdm(total = len(sun_dataloader))
loss_list = []
total_correct = 0
total = 0
for batch, (X, y) in enumerate(sun_dataloader):
    imgs = X.to(device)
    y_hat = model(imgs)
    y = y.type(torch.LongTensor).to(device)
    loss = loss_fn(y_hat, y)
    correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
    total_correct += correct
    total += len(y)
    loss_list.append(loss.item())
    pbar.set_postfix(accuracy=f"{total_correct/total:>7f}")
    pbar.update(1)
pbar.close()
print(f"total test accuracy: {total_correct/total:>7f}, mean loss: {np.mean(loss_list):>7f}")
# %%
test_datapath_origin = "/home/zht/Datasets/proprioception/testSet_c7_ver_2.hdf5"
print("originset:")
origin_data = load_data(test_datapath_origin)
origin_dataloader = DataLoader(origin_data, batch_size=batch_size, shuffle=False)
pbar = tqdm(total = len(origin_dataloader))
loss_list = []
total_correct = 0
total = 0
for batch, (X, y) in enumerate(origin_dataloader):
    imgs = X.to(device)
    y_hat = model(imgs)
    y = y.type(torch.LongTensor).to(device)
    loss = loss_fn(y_hat, y)
    correct = torch.sum(torch.argmax(y_hat, dim=1) == y)
    total_correct += correct
    total += len(y)
    loss_list.append(loss.item())
    pbar.set_postfix(accuracy=f"{total_correct/total:>7f}")
    pbar.update(1)
pbar.close()
print(f"total test accuracy: {total_correct/total:>7f}, mean loss: {np.mean(loss_list):>7f}")
# %%
