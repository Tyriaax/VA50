from cv2 import transform
from detecto.core import Model, Dataset, DataLoader
from detecto import utils, visualize
import os
from torchvision import transforms
import matplotlib.pyplot as plt

PATH_TO_TRAIN = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "train")) + '/'
PATH_TO_TEST = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "test")) + '/'

def trainModel(PATH_TO_IMAGES, classes = [], nameOfTheModel = "model_weights", numberOfEpoch = 10, PATH_OF_THE_SAVED_MODEL = os.path.abspath(os.path.dirname( __file__ )) ):
  if os.path.exists(PATH_TO_IMAGES) and bool(classes):

    custom_transforms = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(800),
      transforms.RandomRotation(degrees=180),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomVerticalFlip(p=0.5),
      transforms.ColorJitter(saturation=0.3),
      transforms.ToTensor(),
      utils.normalize_transform(),
    ])

    dataset = Dataset(PATH_TO_IMAGES, transform = transforms)
    loader = DataLoader(dataset,batch_size = 2, shuffle = true)
    model = Model(classes)
    losses = model.fit(loader, epochs=numberOfEpoch, verbose = True,  )

    plt.plot(losses)  # Visualize loss throughout training
    plt.show()
    #A voir ce qu'on veut en faire et comment les utiliser au mieux

    model.save(PATH_OF_THE_SAVED_MODEL + '/' + nameOfTheModel + ".pth")
  else:
    print("Check the path to the Data or if there is enough classes to train over")

def loadModel(PATH_OF_THE_SAVED_MODEL, classes = []): #Classes must in the same order as initially passed to trainModel()
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

def startLiveRecord(model, scoreFilter = 0.8): #Given model must be running on GPU
  visualize.detect_live(model, score_filter = scoreFilter)

def detectOnVideo(model, PATH_OF_THE_VIDEO, outputFileName, framePerSecond = 30, scoreFilter = 0.8):
  visualize.detect_video(model, PATH_OF_THE_VIDEO, output_file= outputFileName, fps = framePerSecond, score_filter=scoreFilter)


pathLouis = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "TestUnique")) + '/'
pathVideo = os.path.abspath(os.path.dirname( __file__ )) 

model = loadModel(PATH_OF_THE_SAVED_MODEL = 'model_weights.pth', classes = ['mask','nomask'])

#predictImages(model,PATH_TO_TEST)
#startLiveRecord(model)

#detectOnVideo(model, pathVideo,'output_vid.avi')



