import cv2
import os
from enum import Enum
import numpy as np
import random
from Jack import *

class Cards(Enum):
  CBlack = 0
  CBlue = 1
  CBrown = 2
  CGreen = 3
  COrange = 4
  CPink = 5
  CPurple = 6
  CWhite = 7
  CYellow = 8

class ActionPawns(Enum):
  APSherlock = 0
  APAlibi = 1
  APToby = 2
  APWatson = 3
  APJoker = 4
  APReturn = 5
  APChangeCard = 6
  APReturn2 = 7

class DetectivePawns(Enum):
  DPSherlock = 0
  DPToby = 1
  DPWatson = 2

class GameStates(Enum):
  GSWaitingCards = 0
  GSWaitingActionPawnsThrow = 1
  GSUsingActionPawns = 2
  GSAppealOfWitness = 3
  GSGameOver = 4

class GameBoard():
  def __init__(self) -> None:
      self.previousCards = list()
      self.cards = list()

      self.previousCardsState = list()
      self.cardsState = list()

      self.previousDetectivePawns = list()
      self.detective_pawns = list()

      self.action_pawns = [0,0,0,0]
      self.state = GameStates.GSWaitingCards
      self.alibiCardsDict = [
        ("Sgt Goodley", 0, "CBlack"),
        ("Insp. Lestrade", 0, "CBlue"),
        ("Joseph Lane", 1, "CBrown" ),
        ("Miss Stealthy", 1, "CGreen"),
        ("Jeremy Bert", 1, "COrange"),
        ("Madame", 2, "CPink"),        
        ("William Gull", 1, "CPurple"),
        ("John Pizer", 1, "CWhite"),
        ("John Smith", 1, "CYellow"),       
      ]

      self.turnCount = 1
      self.maxTurnCount = 8
      self.jackHourglasses = 0
      self.stage = "Manhunt"
      self.jackWins = False
      self.detectiveWins = False
      self.currentPlayer = "Detective"
      self.jack = self.selectRandomJack()
      self.jack_ai = JackAi() 
      self.isJackFirst = False
      self.actionPawnsPlayed = 0
      self.isJackSeen = False

      self.innocentCards = list()
      self.iaAction = None
      self.actionPawnsNextTurn = None
  
  def getPreviousCards(self):
    return self.previousCards

  def LaunchGame(self):
    self.manhunt()

  def getCards(self):
    return self.cards

  def setCards(self,cards):
    if not len(self.previousCards):
      self.previousCards = self.cards
    self.cards = cards
  
  def getPreviousCardsState(self):
    return self.previousCardsState

  def getCardsState(self):
    return self.cardsState
  
  def setCardsState(self, cardState):
    if not len(self.previousCardsState):
      self.previousCardsState = self.cardsState
    self.cardsState = cardState

  def getPreviousDetectivePawns(self):
    return self.previousDetectivePawns

  def getDetectivePawns(self):
    return self.detective_pawns

  def setDetectivePawns(self,detective_pawns):
    if not len(self.previousDetectivePawns):
      self.previousDetectivePawns = detective_pawns
    self.detective_pawns = detective_pawns

  def getActionPawns(self):
    return self.action_pawns

  def setActionPawns(self, action_pawns):
    self.action_pawns = action_pawns

  def getAlibiCardsDict(self):
    return self.alibiCardsDict
  
  def setAlibiCardsDict(self, alibiCardDict):
    self.alibiCardsDict = alibiCardDict

  def getGameStatus(self):
    return self.state
  
  def getTurnCount(self):
    return self.turnCount
  
  def getMaxTurnCount(self):
    return self.maxTurnCount
  
  def getNumberOfSuspects(self):
    return self.numberOfSuspects
  
  def getJackHourglasses(self):
    return self.jackHourglasses
  
  def getGameStage(self):
    return self.gameStage

  def getJackWins(self):
    return self.jackWins

  def getCurrentPlayer(self):
    return self.currentPlayer

  def getDetectiveWins(self):
    return self.detectiveWins
  
  def tryUpdateGameStatus(self, gameState):
    canUpdate = self.canUpdateGameStatus(gameState)
    if canUpdate:
      self.state = gameState

    return canUpdate

  def canUpdateGameStatus(self, gameState):
    if (((gameState.value < len(GameStates)) and ((gameState == self.state) or (gameState.value == self.state.value + 1)))
    or ((gameState == GameStates.GSWaitingActionPawnsThrow) and self.state == (GameStates.GSAppealOfWitness))):
      return True
    else:
      return False

  def getJackPos(self):

    print("Jack is :", self.jack)
    if self.jack in self.cards:
       index = self.cards.index(self.jack)
       #x, y = index//3 + 1, index%3 + 1
    return index

  def getDetectivesPos(self):
    detectivesPawns = ['DPSherlock', 'DPWatson', 'DPToby']
    correspondingIndexes = ((0,1), (0,2), (0,3), (1,4), (2,4), (3,4), (4,3), (4,2), (4,1), (3,0), (2,0), (1,0))
    detectivesPawnsIndexs = list()

    for detectivePawn in detectivesPawns:
      if detectivePawn in self.detective_pawns:
        detectivesPawnsIndexs.append(correspondingIndexes[self.detective_pawns.index(detectivePawn)])

    return detectivesPawnsIndexs
  
  def updatePreviousCards(self):
    self.previousCards = self.cards
    self.previousCardsState = self.cardsState
  
  def updatePreviousPawnsState(self):
    self.previousDetectivePawns = self.detective_pawns

  def selectRandomJack(self):
    random.shuffle(self.alibiCardsDict)
    jack = self.alibiCardsDict.pop()
    print("Jack is : ", jack[2])
    return jack[2]

  def get_detective_pawn_index(self, list_dp, detective_pawn):
    index_detective = None
    try:
      index_detective = list_dp.index(detective_pawn)
    except:
      for index, element in enumerate(list_dp):
        if type(element) == type(list()) and detective_pawn in element:
          index_detective = index
          break
    return index_detective

  def getIndexCardsChanged(self):
    indexs = []
    if self.cards:
      for index in range(len(self.cards)):
        if ((self.previousCards[index] != self.cards[index]) or (not(np.array_equal(self.previousCardsState[index], self.cardsState[index])))):
          indexs.append(index)
    
    return indexs

  def IsActionPawnRespected(self, action: str):
 
    if action in ["APJoker", "APSherlock", "APToby", "APWatson"]:
      lengthDetectivePawnsList = len(self.detective_pawns)
      indexWatson, previousIndexWatson, indexToby, previousIndexToby, indexSherlock, previousIndexSherlock = (None,)*6
      
      previousIndexSherlock = self.get_detective_pawn_index(self.previousDetectivePawns, "DPSherlock")
      indexSherlock = self.get_detective_pawn_index(self.detective_pawns, "DPSherlock")

      previousIndexToby = self.get_detective_pawn_index(self.previousDetectivePawns, "DPToby")
      indexToby = self.get_detective_pawn_index(self.detective_pawns, "DPToby")

      previousIndexWatson = self.get_detective_pawn_index(self.previousDetectivePawns, "DPWatson")
      indexWatson = self.get_detective_pawn_index(self.detective_pawns, "DPWatson")

      if None not in [indexWatson, previousIndexWatson, indexToby, previousIndexToby, indexSherlock, previousIndexSherlock] and lengthDetectivePawnsList > 0:
        if action == "APJoker":
          if self.currentPlayer == "Jack":
            move_number_watson, move_number_toby, move_number_sherlock = 0, 0, 0
            if self.iaAction[1][0] == "DPWatson":
              move_number_watson = self.iaAction[1][1]
            elif self.iaAction[1][0] == "DPSherlock":
              move_number_sherlock = self.iaAction[1][1]
            elif self.iaAction[1][0] == "DPToby":
              move_number_toby = self.iaAction[1][1]

            if ((previousIndexSherlock + move_number_sherlock) % lengthDetectivePawnsList == indexSherlock ) and \
              ((previousIndexToby + move_number_toby) % lengthDetectivePawnsList == indexToby) and \
              ((previousIndexWatson + move_number_watson) % lengthDetectivePawnsList == indexWatson):
              return True

          elif self.currentPlayer == "Detective":
            if ((previousIndexSherlock + 1) % lengthDetectivePawnsList == indexSherlock and previousIndexToby == indexToby and previousIndexWatson == indexWatson) or \
              ((previousIndexToby + 1) % lengthDetectivePawnsList == indexToby and previousIndexSherlock == indexSherlock and previousIndexWatson == indexWatson) or \
              ((previousIndexWatson + 1) % lengthDetectivePawnsList == indexWatson and previousIndexSherlock == indexSherlock and previousIndexToby == indexToby):
              return True

        elif action == "APSherlock":
          if self.currentPlayer == "Jack":
            move_number_sherlock = self.iaAction[1][1] 
            if (previousIndexSherlock + move_number_sherlock) % lengthDetectivePawnsList == indexSherlock and (previousIndexToby == indexToby and previousIndexWatson == indexWatson):
              return True
          else:
            if (previousIndexSherlock + 1) % lengthDetectivePawnsList == indexSherlock or (
                previousIndexSherlock + 2) % lengthDetectivePawnsList == indexSherlock and (previousIndexToby == indexToby and previousIndexWatson == indexWatson):
              return True

        elif action == "APToby":
          if self.currentPlayer == "Jack":
            move_number_toby = self.iaAction[1][1]
            if (previousIndexToby + move_number_toby) % lengthDetectivePawnsList == indexToby and (
                    previousIndexSherlock == indexSherlock and previousIndexWatson == indexWatson):
              return True
          else:
            if (previousIndexToby + 1) % lengthDetectivePawnsList == indexToby or (
                  previousIndexToby + 2) % lengthDetectivePawnsList == indexToby and (previousIndexSherlock == indexSherlock and previousIndexWatson == indexWatson):
              return True

        elif action == "APWatson":
          if self.currentPlayer == "Jack":
            move_number_watson = self.iaAction[1][1]
            if (previousIndexWatson + move_number_watson) % lengthDetectivePawnsList == indexWatson and (
                    previousIndexSherlock == indexSherlock and previousIndexToby == indexToby):
              return True
          else:
            if (previousIndexWatson + 1) % lengthDetectivePawnsList == indexWatson or (
                    previousIndexWatson + 2) % lengthDetectivePawnsList == indexWatson and (previousIndexSherlock == indexSherlock and previousIndexToby == indexToby):
              return True

    elif action in ["APReturn", "APReturn2"]:
      indexs = self.getIndexCardsChanged()
      if len(indexs) == 1:
        if self.currentPlayer == "Jack":
          if indexs[0] == self.iaAction[1][0] and self.cardsState[indexs[0]][0] == self.iaAction[1][1] and self.previousCardsState[indexs[0]][1] == self.cardsState[indexs[0]][1]:
            return True
        else:
          if self.previousCardsState[indexs[0]][1] == self.cardsState[indexs[0]][1]:
            return True

    elif action == "APChangeCard":

      indexs = self.getIndexCardsChanged()       
      if len(indexs) == 2:
        if self.currentPlayer == "Jack":
          if indexs[0] in self.iaAction[1] and indexs[1] in self.iaAction[1] and np.array_equal(self.previousCardsState[indexs[0]], self.cardsState[indexs[1]]) and np.array_equal(self.previousCardsState[indexs[1]], self.cardsState[indexs[0]]):
            return True
        else:
          if self.cards[indexs[0]] == self.previousCards[indexs[1]] and self.cards[indexs[1]] == self.previousCards[indexs[0]]:
            if np.array_equal(self.previousCardsState[indexs[0]], self.cardsState[indexs[1]]) and np.array_equal(self.previousCardsState[indexs[1]], self.cardsState[indexs[0]]):
              return True

    elif action == "APAlibi": #TODO Verif quelle soit dans le bon sens aussi
      indexs = self.getIndexCardsChanged()
      if self.currentPlayer == "Detective":
        
        if len(indexs) == 1:
          # We check if the card was not turned and if it is indeed the innocented card
          if self.previousCardsState[indexs[0]][0] == self.cardsState[indexs[0]][0] and self.previousCards[indexs[0]] == self.innocentCards[-1]:
            self.alibiCardsDict.pop()
            return True
          # Special case for brown card
          elif self.cardsState[indexs[0]][0] == "cross" and self.previousCards[indexs[0]] == "CBrown"  and self.previousCards[indexs[0]] in self.innocentCards:
            self.alibiCardsDict.pop()
            return True
          else:
            self.innocentCards.pop()
            return False
        elif len(indexs) == 0:
          # If no card has changed, we ensure the innocented cards is not in the cards detected
          if self.innocentCards[-1] not in self.cards:
            self.alibiCardsDict.pop()
            return True
          else:
            self.innocentCards.pop()
            return False
        else:
          # If more than one card has changed it is automatically not validated
          self.innocentCards.pop()
          return False

      else:
        if len(indexs) == 0:
          return True

    print("current cards :\n ", self.cards , "previous cards:\n ", self.previousCards)
    print("current state :\n ", self.cardsState , "previous state:\n ", self.previousCardsState)
    return False
   
  def get_alibi_card(self):
    randomAlibiCard = self.alibiCardsDict[-1]
    
    if randomAlibiCard:
      if self.currentPlayer == "Jack":
        self.addJackHourglasses(randomAlibiCard[1])
      else:
        print(randomAlibiCard)
        self.addInnocentCards([randomAlibiCard[2]])

  def checkVictory(self):
    
    numberOfSuspects = 9 - len(self.innocentCards)
    if self.turnCount == 8 and numberOfSuspects == 1 and self.jackHourglasses > 5:
      if self.isJackSeen:
        self.detectiveWins = True
      else:
        self.jackWins = True

    elif numberOfSuspects == 1:
      self.detectiveWins = True
    elif self.jackHourglasses >= 6:
      self.jackWins = True
  
  def switchPlayer(self):
    if self.currentPlayer == "Jack":
      self.currentPlayer = "Detective"
    else:
      self.currentPlayer = "Jack"
  
  def getNextPlayerToUseActionsPawns(self):

    self.actionPawnsPlayed += 1

    if self.actionPawnsPlayed == 1:
      self.switchPlayer()
    elif self.actionPawnsPlayed == 3:
      self.switchPlayer()
    elif self.actionPawnsPlayed >= 4:
      self.switchPlayer()
  
  def checkCardsPosition(self):
    if self.turnCount == 1:
      return self.validateCardsInitialPosition()
    else:
      validated = True
      indexs = self.getIndexCardsChanged()
      for index in indexs:
        if not(self.previousCardsState[index][0] == self.cardsState[index][0] and self.previousCards[index] in self.innocentCards) and not(self.cardsState[index][0] == "cross" and self.previousCards[index] == "CBrown"  and self.previousCards[index] in self.innocentCards):
          validated = False

      return validated

  def checkPawnsPosition(self):
    if self.turnCount == 1:
      return self.validatePawnsInitialPosition()
    else:
      previousIndexSherlock = self.get_detective_pawn_index(self.previousDetectivePawns, "DPSherlock")
      indexSherlock = self.get_detective_pawn_index(self.detective_pawns, "DPSherlock")

      previousIndexToby = self.get_detective_pawn_index(self.previousDetectivePawns, "DPToby")
      indexToby = self.get_detective_pawn_index(self.detective_pawns, "DPToby")

      previousIndexWatson = self.get_detective_pawn_index(self.previousDetectivePawns, "DPWatson")
      indexWatson = self.get_detective_pawn_index(self.detective_pawns, "DPWatson")

      if previousIndexWatson != indexWatson or previousIndexToby != indexToby or previousIndexSherlock != indexSherlock:
        return False
      elif self.turnCount % 2 == 0:
        return self.checkActionPawnsInverted()
      else:
        return True

  def checkActionPawnsInverted(self):
    for actionPawn in self.action_pawns:
      if actionPawn not in ["APReturn","APReturn2"]:
        if actionPawn not in self.actionPawnsNextTurn:
          return False

    if self.getNumberOfReturnActionPawns(self.action_pawns) == self.getNumberOfReturnActionPawns(self.actionPawnsNextTurn):
      return True
    else:
      return False

  def manhunt(self):
    self.stage = "Manhunt"
    self.actionPawnsPlayed = 0

    if self.turnCount % 2 == 0: 
      self.currentPlayer = "Jack"
      self.isJackFirst = True
    else:
      self.currentPlayer = "Detective"
      self.isJackFirst = False
    
    self.checkVictory()
  
  def addJackHourglasses(self, numberOfHourglasses):
    self.jackHourglasses += numberOfHourglasses

  def appealOfWitnesses(self, isJackSeen):
    self.stage = "Appeal Of Witnesses"
    self.isJackSeen = isJackSeen
    if isJackSeen:
      print("Turn all the cards that are not in sight to their empty side")
      print("Detective wins this turn and get the round token !")
    else:
      self.jackHourglasses += 1
      print("Turn all the cards in sight to their empty side")
      print("Jack wins this turn and get the round token !")

    self.turnCount += 1

  def addInnocentCards(self, innocentCards):
    for i in range(len(innocentCards)):
      if (innocentCards[i] != 0) and (innocentCards[i] not in self.innocentCards):
        self.innocentCards.append(innocentCards[i])

  def getInnocentCards(self):
    return self.innocentCards

  def getInnocentedCard(self):
    return self.innocentCards[len(self.innocentCards)-1]

  def getInnocentCardsIndex(self):
    indexes = []
    for i in range(len(self.innocentCards)):
      indexes.append(Cards[self.innocentCards[i]].value)

    indexes.sort()

    return indexes

  def trySetActionPawnsForNextTurn(self):
    if self.turnCount % 2 == 1 and self.actionPawnsPlayed == 0:
      self.actionPawnsNextTurn = self.getInvertActionPawns(self.action_pawns)
  
  def getInvertActionPawns(self, actionPawnsList):
    actionPawnsNextTurn = list()
    
    for i in range(len(actionPawnsList)):
      actionPawn = ActionPawns[actionPawnsList[i]]
      actionPawnInvert = None
      if actionPawn.value <= 3:
        if actionPawn.value % 2 == 1:
          actionPawnInvert = ActionPawns(actionPawn.value-1).name
        else:
          actionPawnInvert = ActionPawns(actionPawn.value +1).name
      else:
        if actionPawn == ActionPawns.APJoker:
          actionPawnInvert = ActionPawns.APReturn2.name
        elif actionPawn == ActionPawns.APChangeCard:
          actionPawnInvert = ActionPawns.APReturn.name

      if actionPawnInvert not in actionPawnsNextTurn and actionPawnInvert is not None:
        actionPawnsNextTurn.append(actionPawnInvert)

    numberOfReturn = self.getNumberOfReturnActionPawns(self.action_pawns)
    if numberOfReturn == 2:
      actionPawnsNextTurn.append(ActionPawns.APJoker.name)
      actionPawnsNextTurn.append(ActionPawns.APChangeCard.name)
    elif numberOfReturn == 1:
      if ActionPawns.APJoker.name in self.action_pawns:
        actionPawnsNextTurn.append(ActionPawns.APChangeCard.name)
      else:
        actionPawnsNextTurn.append(ActionPawns.APJoker.name)

    return actionPawnsNextTurn

  def getNumberOfReturnActionPawns(self, actionPawnsList):
    numberOfReturn = 0
    for i in range(len(actionPawnsList)):
      if (actionPawnsList[i] in ["APReturn", "APReturn2"]):
        numberOfReturn += 1

    return numberOfReturn

  def nextTurn(self):
    self.getNextPlayerToUseActionsPawns()
    if self.currentPlayer == "Jack" and self.actionPawnsPlayed < 4:
      self.jackPlays()
    else:
      self.iaAction = None


  def tryComputeIaAction(self):
    if self.isJackFirst:
      self.jackPlays()

  def jackPlays(self):

    game_board = {
      "cardsPosition" : self.cards,
      "cardsOrientation" : self.cardsState,
      "dectectivePawns" : self.detective_pawns,
      "hourglasses" : self.jackHourglasses,
      "jack" : self.jack,
      "remaining_suspect" : len(self.cards),
      "remaining_card_suspect" : self.alibiCardsDict
    }   
  
    self.iaAction  = self.jack_ai.jack(game_board, self.actionPawnsPlayed, self.isJackFirst, self.getActionPawns())
    if self.iaAction[0] == "APAlibi": #if the ia picks alibi
      self.get_alibi_card()
    print(self.iaAction)

  def getIaAction(self):
    return self.iaAction

  def validateCardsInitialPosition(self):
    validated = True

    for cards in self.cardsState:
      if cards[1] != "front":
        validated = False

    if self.cardsState[0][0] != "right":
      validated = False

    if self.cardsState[2][0] != "left":
      validated = False

    if self.cardsState[7][0] != "up":
      validated = False

    return validated

  def validatePawnsInitialPosition(self):
    validated = True

    if self.detective_pawns[3] != "DPWatson":
      validated = False

    if self.detective_pawns[7] != "DPToby":
      validated = False

    if self.detective_pawns[11] != "DPSherlock":
      validated = False

    return validated
