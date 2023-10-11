import torch 
from torch import nn

# model architecture
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

