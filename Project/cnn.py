import cv2
import numpy as np
import torch
from torchvision import transforms

class cnnHelper:
    def __init__(self, type):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        CNN = torch.load('CNN/' + type + '_RECOGNITION_CNN100epochs.pt')
        self.CNN = CNN.to(self.device)

        self.CNNPreprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def ComputeImage(self, img, resizeDimInCircle = 0):
        if resizeDimInCircle != 0:
            # If we put a value here we will resize the image in square to crop with a circle mask
            dim = (resizeDimInCircle, resizeDimInCircle)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
            height, width = img.shape[:2]
            mask = np.full((height, width), 0, dtype=np.uint8)
            cv2.circle(mask, (height // 2, width // 2), height // 2, 255, -1)

            img = cv2.bitwise_and(img, img, mask=mask)

        imgtensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input = self.CNNPreprocess(imgtensor)
        input.unsqueeze_(0)
        input = input.to(self.device)

        # Here we format the output data to retrieve probabilities
        output = self.CNN(input)
        sm = torch.nn.Softmax(dim=1)
        probability = sm(output)
        probability = probability.cpu().detach().numpy()[0]
        probability = probability.tolist()

        return probability