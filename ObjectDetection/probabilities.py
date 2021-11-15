from drawing import *

def combineProbabilities(probabilitiesList,weights):
  numberOfProbabilitiesToCombine = len(probabilitiesList)
  numberOfObjects = len(probabilitiesList[0])
  numberOfSamples = len(probabilitiesList[0][0])

  combinedProbability = [[0 for i in range(numberOfSamples)] for j in range(numberOfObjects)]

  for i in range(numberOfProbabilitiesToCombine):
    for j in range(numberOfObjects):
      for k in range(numberOfSamples):
        combinedProbability[j][k] = combinedProbability[j][k] + probabilitiesList[i][j][k]*weights[i]

  for i in range(numberOfObjects):
    sumValue = sum(combinedProbability[i])
    for j in range(numberOfSamples):
      combinedProbability[i][j] = combinedProbability[i][j]/sumValue

  return combinedProbability

def drawRectangleWithProbabilities(img,probabilities,boundingBoxes,alreadydetectedobjects,enum):
  minProbability = 0

  maxproba = []
  for i in range(len(probabilities)):
    maxproba.append(max(probabilities[i]))

  maxValueBb = max(maxproba)
  indexMaxValueBb = maxproba.index(maxValueBb)
  maxValue = max(probabilities[indexMaxValueBb])
  if (maxValue > minProbability):
    indexMaxValue = probabilities[indexMaxValueBb ].index(maxValue)

    if indexMaxValue not in alreadydetectedobjects:
      alreadydetectedobjects.append(indexMaxValue)
      img = drawRectangle(img, boundingBoxes[indexMaxValueBb], enum(indexMaxValue).name)
      boundingBoxes.remove(boundingBoxes[indexMaxValueBb])
      probabilities.remove(probabilities[indexMaxValueBb])
    else:
      probabilities[indexMaxValueBb][indexMaxValue] = 0

    if(len(boundingBoxes) > 0):
      img = drawRectangleWithProbabilities(img, probabilities, boundingBoxes,alreadydetectedobjects,enum)

  return img