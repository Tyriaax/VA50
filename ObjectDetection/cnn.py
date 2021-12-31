import torch
from torchvision import transforms
import cv2

class cnnHelper:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        CNN = torch.load('CNN/CNN/AP_RECOGNITION_CNN.pt')
        self.CNN = CNN.to(self.device)

        self.CNNPreprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def ComputeImage(self, img):
        imgtensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input = self.CNNPreprocess(imgtensor)
        input.unsqueeze_(0)
        input = input.to(self.device)

        output = self.CNN(input)
        sm = torch.nn.Softmax(dim=1)
        probability = sm(output)
        probability = probability.cpu().detach().numpy()[0]
        probability = probability.tolist()

        return probability