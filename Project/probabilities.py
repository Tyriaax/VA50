from scipy.optimize import linear_sum_assignment
from drawing import *

# This function is made to combine multiple array of probabilities using the weight array
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

# This function is made to use the linear assignment algorithm
# It formats the probabilities array in a cost matrix, while inverting the probability
# so that the cost is inversely proportional
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

# There is a same version using strings array instead of enum directly
def linearAssignmentWithStrings(finalProbabilities, selectedStrings):
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
    result.append(selectedStrings[col_ind[i]])

  return result

# This fonction duplicates the probabilities of recognition for the return action pawn since there is only 7 classes in the cnn but 8 classes (2 return pawns)
def FormatActionPawnProbabilitiesMissingSample(probabilities):
  for probability in probabilities:
    probability.append(probability[5])
    probabilitiessum = sum(probability)
    probability[:] = [prob / probabilitiessum for prob in probability]

  return probabilities