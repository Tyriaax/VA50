from tkinter.constants import W
from cv2 import transform
from detecto.core import Model, Dataset, DataLoader
from detecto import utils, visualize
import os
from torch._C import Size
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import cv2

#Dossier contenant les imgs et label devant obligatoirement s'appeller comme ça

def trainModel(PATH_TO_DATAS : str, classes = [], nameOfTheModel = "model_weights", numberOfEpoch = 25, PATH_OF_THE_SAVED_MODEL = os.path.abspath(os.path.dirname( __file__ ))) -> None:

  if os.path.exists(PATH_TO_DATAS) and bool(classes) and os.path.exists(PATH_OF_THE_SAVED_MODEL):

    PATH_TO_TRAIN_DATA_CSV = os.path.join(PATH_TO_DATAS, "train_labels.csv")
    PATH_TO_VALIDATE_DATA_CSV = os.path.join(PATH_TO_DATAS, "validation_labels.csv")
    PATH_TO_TRAIN_LABELS = os.path.join(PATH_TO_DATAS, "train/labels")
    PATH_TO_VALIDATE_LABELS = os.path.join(PATH_TO_DATAS, "validation/labels")
    PATH_TO_TRAIN_IMAGES = os.path.join(PATH_TO_DATAS, "train/images")
    PATH_TO_VALIDATE_IMAGES = os.path.join(PATH_TO_DATAS, "validation/images")
    PATH_OF_THE_SAVED_MODEL = os.path.join(PATH_OF_THE_SAVED_MODEL, nameOfTheModel +".pth")

    custom_transforms = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(800),
      transforms.RandomRotation(degrees = 180),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomVerticalFlip(p=0.5),
      transforms.ColorJitter(saturation=0.3),
      transforms.ToTensor(),
      utils.normalize_transform(),
    ])

    utils.xml_to_csv(PATH_TO_TRAIN_LABELS, PATH_TO_TRAIN_DATA_CSV )
    utils.xml_to_csv(PATH_TO_VALIDATE_LABELS, PATH_TO_VALIDATE_DATA_CSV) 

    trainingDataset = Dataset(PATH_TO_TRAIN_DATA_CSV, PATH_TO_TRAIN_IMAGES, transform = custom_transforms)
    loader = DataLoader(trainingDataset, batch_size = 2, shuffle = True) #batch_size : How many images at once ----> A changer !!!!

    validationDataset = Dataset(PATH_TO_VALIDATE_DATA_CSV, PATH_TO_VALIDATE_IMAGES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(classes = classes, device = device)
    losses = model.fit(loader, val_dataset = validationDataset, epochs = numberOfEpoch, verbose = True,  )

    plt.plot(losses)  # Visualize loss throughout training
    plt.show()
    #A voir ce qu'on veut en faire et comment les utiliser au mieux

    model.save(PATH_OF_THE_SAVED_MODEL)
  else:
    print("Check the path to the Data or if there is enough classes to train over")

def loadModel(PATH_OF_THE_SAVED_MODEL : str, classes : list): #Classes must in the same order as initially passed to trainModel()
  if os.path.exists(PATH_OF_THE_SAVED_MODEL) and PATH_OF_THE_SAVED_MODEL.lower().endswith(".pth") and bool(classes):
    model = Model.load(PATH_OF_THE_SAVED_MODEL, classes)
    return model
  else:
    print("- Check the path to the saved model : He must be a .pth file\n- Classes must in the same order as initially passed to trainModel() ")

def predictImages(model, PATH_TO_TEST_IMAGES): #Mettre un nom
  images = []
  for file in os.listdir(PATH_TO_TEST_IMAGES):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
      image = utils.read_image(PATH_TO_TEST_IMAGES + file)
      images.append(image)

  if len(images) > 1 :
    visualize.plot_prediction_grid(model, images, dim = (len(images), 1), score_filter = 0.8, figsize = (30,30))
  elif len(images) == 1:
    labels,boxes,scores = model.predict(images[0])
    print(labels)
    print(boxes)
    print(scores)
    visualize.show_labeled_image(images[0], boxes[0], labels[0])

def startLiveRecord(model, scoreFilter = 0.25): #Given model must be running on GPU

  cv2.namedWindow('Detecto')
  try:
    video = cv2.VideoCapture(0)
  except:
    print('No webcam available.')
    return

  while True:
    ret, frame = video.read()
    if not ret:
      break

    labels, boxes, scores = model.predict(frame)
# Plot each box with its label and score
    for i in range(boxes.shape[0]):
      if scores[i] < scoreFilter:
        continue
      
      box = boxes[i]
      print(box, type(box))
      print(f"labels : {labels}, boxes : {boxes}, scores : {scores}")
      cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
      print("lets go")
      if labels:
        cv2.putText(frame, '{}: {}'.format(labels[i], round(scores[i].item(), 2)), (int(box[0]), int(box[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Detecto", frame)

        # If the 'q' or ESC key is pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
      break       

  video.release()
  cv2.destroyAllWindows()
 

  #visualize.detect_live(model, score_filter = scoreFilter)
  #print("press 'q' or 'escape' to quit the live detection")

def detectOnVideo(model, PATH_OF_THE_VIDEO, outputFileName, framePerSecond = 30, scoreFilter = 0.1) :
  visualize.detect_video(model, PATH_OF_THE_VIDEO, output_file= outputFileName, fps = framePerSecond, score_filter=scoreFilter)

import xml.etree.ElementTree as ET

def verify_XML_IMG(PATH_TO_DATAS):
  
  PATH_TO_DATAS = os.path.join(PATH_TO_DATAS, "train")

  files = {x: os.listdir(os.path.join(PATH_TO_DATAS, x)) for x in ['images', 'labels']}
  dataset_sizes = {x: len(files[x]) for x in ['images', 'labels']}

  for index in range(dataset_sizes['images']):
    tree = ET.parse(os.path.join(PATH_TO_DATAS,"labels", files['labels'][index]))
    root = tree.getroot()

    for member in root.findall('object'):
      box = member.find('bndbox')
      label = member.find('name').text

      sp = (int(float(box.find('xmin').text)), int(float(box.find('ymin').text)))
      ep = (int(float(box.find('xmax').text)), int(float(box.find('ymax').text)))
      image = cv2.imread(os.path.join(PATH_TO_DATAS,"images", files['images'][index]))
      window_name = "image"

      cv2.rectangle(image, sp, ep, (255, 0, 0), 3)
      cv2.imshow(window_name, image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()




PATH_TO_DATAS = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "Square")) 
PATH_OF_THE_SAVED_MODEL = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "Square_weights.pth")) 

#trainModel(PATH_TO_DATAS , classes = ['Square'], nameOfTheModel = "Square_weights", numberOfEpoch = 25, PATH_OF_THE_SAVED_MODEL = os.path.abspath(os.path.dirname( __file__ )))

model = loadModel(PATH_OF_THE_SAVED_MODEL = PATH_OF_THE_SAVED_MODEL, classes = ['Square'])

#predictImages(model,PATH_TO_TEST)
startLiveRecord(model)

#detectOnVideo(model, os.path.abspath(os.path.join(os.path.dirname( __file__ ),"carré.mp4")), os.path.abspath(os.path.join(os.path.dirname( __file__ ), "output_vid.avi"))) 

#verify_XML_IMG(PATH_TO_DATAS)



