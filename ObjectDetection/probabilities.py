import numpy as np

from drawing import *
from scipy.optimize import linear_sum_assignment

def combineProbabilities(probabilitiesList,weights):
  numberOfProbabilitiesToCombine = len(probabilitiesList)
  numberOfObjects = len(probabilitiesList[0])
  numberOfSamples = len(probabilitiesList[0][0])

  combinedProbability = [[0 for i in range(numberOfSamples)] for j in range(numberOfObjects)]

  for i in range(numberOfProbabilitiesToCombine):
    for j in range(numberOfObjects):
      for k in range(numberOfSamples):
        combinedProbability[j][k] = combinedProbability[j][k] + probabilitiesList[i][j][k]*weights[i]

  return combinedProbability

def drawRectangleWithProbabilities(img,probabilities,boundingBoxes,enum, cardBoard):
  minProbability = 0

  maxproba = []
  for i in range(len(probabilities)):
    maxproba.append(max(probabilities[i]))

  maxValueBb = max(maxproba)

  if(maxValueBb > minProbability):
    indexMaxValueBb = maxproba.index(maxValueBb)
    
    maxValue = max(probabilities[indexMaxValueBb])

    indexMaxValue = probabilities[indexMaxValueBb ].index(maxValue)

    img = drawRectangle(img, boundingBoxes[indexMaxValueBb], enum(indexMaxValue).name)
    cardBoard[indexMaxValueBb] = enum(indexMaxValue).name
    probabilities[indexMaxValueBb] = [0 for i in range(len(probabilities[indexMaxValueBb]))]
    for i in range(len(probabilities)):
      probabilities[i][indexMaxValue] = 0

    img = drawRectangleWithProbabilities(img, probabilities, boundingBoxes,enum, cardBoard)

  return img

def linearAssignment(finalProbabilities, selectedEnum):
  costmatrix = np.zeros((len(finalProbabilities),len(finalProbabilities[0])))
  for i in range(len(finalProbabilities)):
    array = np.array(finalProbabilities[i])
    if np.isnan(array).any():
      array = np.full(len(array),1000000)
    else:
      for j in range(len(array)):
        if array[j] != 0:
          array[j] = 1/array[j]
        else:
          array[j] = 1000000

    costmatrix[i] = array

  row_ind, col_ind = linear_sum_assignment(costmatrix)

  result = []
  for i in range(len(col_ind)):
    result.append(selectedEnum(col_ind[i]).name)

  return result

def FormatActionPawnProbabilitiesMissingSample(probabilities):
  for probability in probabilities:
    probability.append(probability[5])
    probabilitiessum = sum(probability)
    probability[:] = [prob / probabilitiessum for prob in probability]

  return probabilities