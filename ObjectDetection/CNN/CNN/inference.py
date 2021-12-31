import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random, os

net = torch.load('ANIMEMOVIE_CNN.pt')
net = net.cuda()

label = ['Anime','Movie']

path = "./data/val/"
type = random.randint(0,1)
if type == 0:
    path += "anime/"
else:
    path += "movie/"
imgpath = random.choice(os.listdir(path))
img = Image.open(path+imgpath)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

input = preprocess(img)
input.unsqueeze_(0)
input = input.cuda()

output = net(input)
score, preds = torch.max(output,1)
plt.imshow(img)
plt.title("Prediction : " + label[preds[0]])
plt.show()