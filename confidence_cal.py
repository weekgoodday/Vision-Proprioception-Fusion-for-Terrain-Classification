#%%
from temperature_scaling import ModelWithTemperature
import torch
from torch import nn
import h5py
from torchvision import transforms
from dataset import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np 
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os

#hyperparameters
batch_size = 64
temperature = 8

# model architecture
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')
for child in model.named_children():
    print(child)
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
model.load_state_dict(torch.load("./model/model997.pth"))
model.eval()

# prepare data
valid_datapath = "/home/zht/Datasets/proprioception/validSet_c7_corrected.hdf5"
test_datapath = "/home/zht/Datasets/proprioception/testSet_c7_ver_2.hdf5"
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
print("validset:")
valid_data = load_data(valid_datapath)
print("testset:")
test_data = load_data(test_datapath)
dark_dataloader = DataLoader(test_data1, batch_size=batch_size, shuffle=False)
fog_dataloader = DataLoader(test_data2, batch_size=batch_size, shuffle=False)
sun_dataloader = DataLoader(test_data3, batch_size=batch_size, shuffle=False)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
orig_model = model
# %%
# temperature scaling
scaled_model = ModelWithTemperature(orig_model, t=temperature) # specify temperature
# scaled_model.set_temperature(sun_dataloader)  # scaling dataset can be changed here, but optimize on dataset will get confusing temperature, so just designate temperature in the first place
model = scaled_model
with open("./reliability_diagrams/record.txt", "a+") as f:
    f.write(f"temperature: {temperature}\n")
# %%
# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# just convient for name output
class CallingCounter:
    def __init__(self, func):
        self.count = 0
        self.func = func
    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

# plot reliability diagram
def plot_lines(bins_accuracy):
    n_bins = bins_accuracy.shape[0]
    x=np.arange(0,1,1/n_bins)+1/n_bins/2
    y=bins_accuracy
    fig,ax=plt.subplots(figsize=(8,8))
    tick=np.arange(0,1.1,0.1)
    plt.plot(x,y,color='red',alpha=1,linestyle="--",linewidth=2,marker="o",markersize=20)
    line,=plt.plot([0,1],[0,1],ls='--',color='grey')
    line.set_dashes((3,7))
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    plt.grid(True,color='black',linewidth=0.5,linestyle='--',dashes=(5,15))
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    i = 0
    name = ['test', 'dark', 'fog', 'sun']
    while os.path.exists(f"./reliability_diagrams/{name[i % 4]}_line_{temperature}.jpg"):
        i += 1
    fig.savefig(f'./reliability_diagrams/{name[i % 4]}_line_{temperature}.jpg', bbox_inches='tight')
    # plt.show()
def plot_diagrams(bins_props_in):
    n_bins = bins_props_in.shape[0]
    x=np.arange(0,1,1.0/n_bins)
    accuracy=bins_props_in
    fig,ax=plt.subplots(figsize=(8,8))
    tick=np.arange(0,1.1,1.0/n_bins)
    plt.bar(x,accuracy,width=0.1,align='edge',color='blue',alpha=0.4,edgecolor='black',linewidth=0.5)
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    plt.grid(True,color='black',linewidth=0.5,linestyle='--',dashes=(5,15))
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel('Confidence')
    plt.ylabel('Percentage')
    # plt.show()
    i = 0
    name = ['test', 'dark', 'fog', 'sun']
    while os.path.exists(f"./reliability_diagrams/{name[i % 4]}_bar_{temperature}.jpg"):
        i += 1
    fig.savefig(f'./reliability_diagrams/{name[i % 4]}_bar_{temperature}.jpg', bbox_inches='tight')
@CallingCounter # just count how many times this function is called
def calculate_diagram(dataloader, n_bins = 10):
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for input, label in dataloader:
            input = input.cuda()
            label = label.type(torch.LongTensor)
            logits = model(input)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1) # return logits' max value and index, 1 means row
    accuracies = predictions.eq(labels)
    print("total accuracy", accuracies.float().mean())
    ece = torch.zeros(1, device=logits.device)
    bins_accuracy = torch.zeros(n_bins, device=logits.device)
    bins_confidence = torch.zeros(n_bins, device=logits.device)
    bins_prop_in = torch.zeros(n_bins, device=logits.device)
    for bin_num, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bins = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item()) # gt: greater than, le: less than or equal to, return 0 or 1,
        # final return the index of samples belong to this bin
        prop_in_bin = in_bins.float().mean() # in_bin numbers / total numbers
        bins_prop_in[bin_num] = prop_in_bin
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bins].float().mean()
            avg_confidence_in_bin = confidences[in_bins].mean()
            bins_accuracy[bin_num] = accuracy_in_bin
            bins_confidence[bin_num] = avg_confidence_in_bin
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            print(f"no samples in bin [{bin_lower}, {bin_upper}]")
    name = ['test', 'dark', 'fog', 'sun']
    print(f"On {name[calculate_diagram.count - 1]} set:")
    print(f"ECE: {ece.item():.3f}")
    print("bins accuracy", bins_accuracy)
    print("bins confidence", bins_confidence)
    print("bins prop in", bins_prop_in)
    with open("./reliability_diagrams/record.txt", "a+") as f:
        f.write(f"On {name[calculate_diagram.count - 1]} set: ECE: {ece.item():.3f}\n")
    plot_lines(bins_accuracy.cpu().numpy())
    plot_diagrams(bins_prop_in.cpu().numpy())
calculate_diagram(test_dataloader)
calculate_diagram(dark_dataloader)
calculate_diagram(fog_dataloader)
calculate_diagram(sun_dataloader)

# %%
