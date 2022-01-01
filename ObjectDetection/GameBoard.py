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
        ("Joseph Lane", 1, "CBrown" ),
        ("Madame", 2, "CPink"),
        ("Insp. Lestrade", 0, "CBlue"),
        ("William Gull", 1, "CPurple"),
        ("Jeremy Bert", 1, "COrange"),
        ("John Smith", 1, "CYellow"),
        ("Sgt Goodley", 0, "CBlack"),
        ("Miss Stealthy", 1, "CGreen"),
        ("John Pizer", 1, "CWhite"),
      ]

      self.turnCount = 1
      self.maxTurnCount = 8
      self.numberOfSuspects = 9
      self.jackHourglasses = 0
      self.stage = "Manhunt"
      self.jackWins = False
      self.detectiveWins = False
      self.currentPlayer = "Detective"
      self.jack = self.selectRandomJack()
      self.jack_ai = JackAi() 
      self.isJackFirst = False
      self.actionPawnsPlayed = 0

      self.innocentCards = list()
      self.iaAction = None
  
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
    randomIndex = random.randint(0, len(self.alibiCardsDict) - 1)
    print("Jack is : ", self.alibiCardsDict[randomIndex])
    return self.alibiCardsDict.pop(randomIndex)[2]

  def IsActionPawnRespected(self, action: str):
 
    if action in ["APJoker", "APSherlock", "APToby", "APWatson"]:
      lengthDetectivePawnsList = len(self.detective_pawns)
      indexWatson, previousIndexWatson, indexToby, previousIndexToby, indexSherlock, previousIndexSherlock = (None,)*6

      if "DPSherlock" in self.previousDetectivePawns and "DPSherlock" in self.detective_pawns:
        previousIndexSherlock = self.previousDetectivePawns.index("DPSherlock")
        indexSherlock = self.detective_pawns.index("DPSherlock")

      if "DPToby" in self.previousDetectivePawns and "DPToby" in self.detective_pawns:
        previousIndexToby = self.previousDetectivePawns.index("DPToby")
        indexToby = self.detective_pawns.index("DPToby")

      if "DPWatson" in self.previousDetectivePawns and "DPWatson" in self.detective_pawns:
        previousIndexWatson = self.previousDetectivePawns.index("DPWatson")
        indexWatson = self.detective_pawns.index("DPWatson")

      if None not in [indexWatson, previousIndexWatson, indexToby, previousIndexToby, indexSherlock, previousIndexSherlock] and lengthDetectivePawnsList > 0:
        if action == "APJoker":
          if self.currentPlayer == "Jack":
            if ((previousIndexSherlock + 1) % lengthDetectivePawnsList == indexSherlock and previousIndexToby == indexToby and previousIndexWatson == indexWatson) or \
              ((previousIndexToby + 1) % lengthDetectivePawnsList == indexToby and previousIndexSherlock == indexSherlock and previousIndexWatson == indexWatson) or \
              ((previousIndexWatson + 1) % lengthDetectivePawnsList == indexWatson and previousIndexSherlock == indexSherlock and previousIndexToby == indexToby) or \
              (previousIndexToby == indexToby and previousIndexWatson == indexWatson and previousIndexSherlock == indexSherlock) :
              return True

          elif self.currentPlayer == "Detective":
            if ((previousIndexSherlock + 1) % lengthDetectivePawnsList == indexSherlock and previousIndexToby == indexToby and previousIndexWatson == indexWatson) or \
              ((previousIndexToby + 1) % lengthDetectivePawnsList == indexToby and previousIndexSherlock == indexSherlock and previousIndexWatson == indexWatson) or \
              ((previousIndexWatson + 1) % lengthDetectivePawnsList == indexWatson and previousIndexSherlock == indexSherlock and previousIndexToby == indexToby):
              return True

        elif action == "APSherlock":
          if (previousIndexSherlock + 1) % lengthDetectivePawnsList == indexSherlock or (
                  previousIndexSherlock + 2) % lengthDetectivePawnsList == indexSherlock and (previousIndexToby == indexToby and previousIndexWatson == indexWatson):
            return True

        elif action == "APToby":
          if (previousIndexToby + 1) % lengthDetectivePawnsList == indexToby or (
                  previousIndexToby + 2) % lengthDetectivePawnsList == indexToby and (previousIndexSherlock == indexSherlock and previousIndexWatson == indexWatson):
            return True

        elif action == "APWatson":
          if (previousIndexWatson + 1) % lengthDetectivePawnsList == indexWatson or (
                  previousIndexWatson + 2) % lengthDetectivePawnsList == indexWatson and (previousIndexSherlock == indexSherlock and previousIndexToby == indexToby):
            return True

    elif action in ["APReturn", "APReturn2"]:
      difference = 0
      if self.previousCards == self.cards and self.previousCards:
        for index in range(len(self.cardsState)):
          if self.previousCardsState[index][0] != self.cardsState[index][0] and self.previousCardsState[index][1] == self.cardsState[index][1]:
            difference += 1
        if difference <= 1:
          return True

    elif action == "APChangeCard":
      indexs = []
      if self.cards:
        for index in range(len(self.cards)):
          if not np.array_equal(self.previousCards[index], self.cards[index]):
            indexs.append(index)

        if len(indexs) == 2:
          if self.cards[indexs[0]] == self.previousCards[indexs[1]] and self.cards[indexs[1]] == self.previousCards[indexs[0]]:
            if np.array_equal(self.previousCardsState[indexs[0]], self.cardsState[indexs[1]]) and np.array_equal(self.previousCardsState[indexs[1]], self.cardsState[indexs[0]]):
              return True

    elif action == "APAlibi":
      for card in self.cards:
        if card in self.innocentCards:
          return False
      
      return True

    print("current cards :\n ", self.cards , "previous cards:\n ", self.previousCards)
    print("current state :\n ", self.cardsState , "previous state:\n ", self.previousCardsState)
    return False
   
  def get_alibi_card(self):
    randomIndex = random.randint(0, len(self.alibiCardsDict) - 1)
    randomAlibiCard = self.alibiCardsDict.pop(randomIndex)
    if randomAlibiCard:
      if self.currentPlayer == "Jack":
        self.addJackHourglasses(randomAlibiCard[1])
      else:
        print(randomAlibiCard)
      self.addInnocentCards([randomAlibiCard[2]])

  def checkVictory(self, isJackSeen):
    if self.stage == "Appeal for Witnesses":
      if self.numberOfSuspects == 1 and self.jackHourglasses == 6:
        if isJackSeen:
          self.detectiveWins = True
        else:
          self.jackWins = True

      elif self.numberOfSuspects == 1 and self.jackHourglasses < 6:
        self.detectiveWins = True
      elif self.numberOfSuspects > 1 and self.jackHourglasses == 6:
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
      self.actionPawnsPlayed = 0

  def manhunt(self):
    self.stage = "Manhunt"

    if self.turnCount % 2 == 0: 
      self.currentPlayer = "Jack"
      self.isJackFirst = True
      print("Flip back the tokens.")
    else:
      self.currentPlayer = "Detective"
      self.isJackFirst = False
      print("Detective starts: you can throw the tokens")
  
  def addJackHourglasses(self, numberOfHourglasses):
    self.jackHourglasses += numberOfHourglasses

  def appealOfWitnesses(self, isJackSeen):
    self.stage = "Appeal Of Witnesses"
    if isJackSeen:
      print("Turn all the cards that are not in sight to their empty side")
      print("Detective wins this turn and get the round token !")
    else:
      self.jackHourglasses += 1
      print("Turn all the cards in sight to their empty side")
      print("Jack wins this turn and get the round token !")
    
    self.checkVictory(isJackSeen)
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

  def nextTurn(self):
    self.getNextPlayerToUseActionsPawns()
    if self.currentPlayer == "Jack":
      self.jackPlays()
    else:
      self.iaAction = None

  def jackPlays(self):

    game_board = {
      "cardsPosition" : self.cards,
      "cardsOrientation" : self.cardsState,
      "dectectivePawns" : self.detective_pawns,
      "hourglasses" : self.jackHourglasses,
      "jack" : self.jack 
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
