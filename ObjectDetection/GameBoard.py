import cv2
import os
from enum import Enum
import numpy as np
import random

class GameStates(Enum):
  GSWaitingHomography = 0
  GSWaitingActionPawns = 1
  GSUseActionsPawns = 2

from cards_recognition import*

class GameBoard():
  def __init__(self) -> None:
      self.previousCards = [0,0,0,0,0,0,0,0,0]
      self.cards = [0,0,0,0,0,0,0,0,0]

      self.previousCardsState = []
      self.cardsState = []

      self.previousDetectivePawns = []
      self.detective_pawns = [0,0,0,0,0,0,0,0,0,0,0,0]

      self.action_pawns = [0,0,0,0]
      self.board_file = os.path.abspath(os.path.join(os.path.dirname( __file__ ),"Game_state","JackPocketBoard.txt"))
      self.state = GameStates.GSWaitingHomography

      self.alibiCardsDict = (
        ("Joseph Lane", 1),
        ("Madame", 2),
        ("Insp. Lestrade", 0),
        ("William Gull", 1),
        ("Jeremy Bert", 1),
        ("John Smith", 1),
        ("Sgt Goodley", 0),
        ("Miss Stealthy", 1),
        ("John Pizer", 1),
      )

      self.turnCount = 1
      self.maxTurnCount = 8
      self.numberOfSuspects = 9
      self.jackHourglasses = 0
      self.stage = "Manhunt"
      self.jackWins = False
      self.detectiveWins = False
      self.currentPlayer = "Detective"
  
  def getPreviousCards(self):
    return self.previousCards

  def getCards(self):
    return self.cards

  def setCards(self,cards):
    self.previousCards = self.cards
    self.cards = cards
  
  def getPreviousCardsState(self):
    return self.previousCardsState

  def getCardsState(self):
    return self.cardsState
  
  def setCardsState(self, cardState):
    self.previousCardsState = self.cardsState
    self.cardsState = cardState

  def getPreviousDetectivePawns(self):
    return self.previousDetectivePawns

  def getDetectivePawns(self):
    return self.detective_pawns

  def setDetectivePawns(self,detective_pawns):
    self.previousDetectivePawns = self.detective_pawns
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
    if ((gameState.value < len(GameStates)) and
            ((gameState == self.state) or (gameState.value == self.state.value+1))):
      self.state = gameState
      return True
    else:
      return False
  
  def IsActionPawnRespected(self, action: str):

    lengthDetectivePawnsList = len(self.detective_pawns)

    if "DPSherlock" in self.previousDetectivePawns and "DPSherlock" in self.detective_pawns:
      previousIndexSherlock = self.previousDetectivePawns.index("DPSherlock")
      indexSherlock = self.detective_pawns.index("DPSherlock")

    if "DPToby" in self.previousDetectivePawns and "DPToby" in self.detective_pawns:
      previousIndexToby = self.previousDetectivePawns.index("DPToby")
      indexToby = self.detective_pawns.index("DPToby")

    if "DPWatson" in self.previousDetectivePawns and "DPWatson" in self.detective_pawns:
      previousIndexWatson = self.previousDetectivePawns.index("DPWatson")
      indexWatson = self.detective_pawns.index("DPWatson")

    if action == "APJoker":
      if self.currentPlayer == "Jack":
        if ((previousIndexSherlock + 1) % lengthDetectivePawnsList == indexSherlock and previousIndexToby == indexToby and previousIndexWatson == indexWatson) or \
          ((previousIndexToby + 1) % lengthDetectivePawnsList == indexToby and previousIndexSherlock == indexSherlock and previousIndexWatson == indexWatson) or \
          ((previousIndexWatson + 1) % lengthDetectivePawnsList == indexWatson and previousIndexSherlock == indexSherlock and previousIndexToby == indexToby) or \
          (previousIndexToby == indexToby and previousIndexWatson == indexWatson and previousIndexSherlock == indexSherlock) :
          return True

      elif self.currentPlayer == "Detectives":
        if ((previousIndexSherlock + 1) % lengthDetectivePawnsList == indexSherlock and previousIndexToby == indexToby and previousIndexWatson == indexWatson) or \
          ((previousIndexToby + 1) % lengthDetectivePawnsList == indexToby and previousIndexSherlock == indexSherlock and previousIndexWatson == indexWatson) or \
          ((previousIndexWatson + 1) % lengthDetectivePawnsList == indexWatson and previousIndexSherlock == indexSherlock and previousIndexToby == indexToby):
          return True

    elif action == "APHolmes":
      if (previousIndexSherlock + 1) % lengthDetectivePawnsList == indexSherlock or (
              previousIndexSherlock + 2) % lengthDetectivePawnsList == indexSherlock:
        return True

    elif action == "APToby":
      if (previousIndexToby + 1) % lengthDetectivePawnsList == indexSherlock or (
              previousIndexToby + 2) % lengthDetectivePawnsList == indexSherlock:
        return True

    elif action == "APWatson":
      if (previousIndexWatson + 1) % lengthDetectivePawnsList == indexSherlock or (
              previousIndexWatson + 2) % lengthDetectivePawnsList == indexSherlock:
        return True

    elif action == "APRotation":
      difference = 0
      if self.previousCards == self.cards and self.previousCards:
        for index in range(len(self.cardsState)):
          if self.previousCardsState[index][0] != self.cardsState[index][0] and self.previousCardsState[index][1] == self.cardsState[index][1]:
            difference += 1
        if difference <= 1:
          return True

    elif action == "APExchange":
      indexs = []
      if self.cards:
        for index in range(len(self.cards)):
          if not np.array_equal(self.previousCards[index], self.cards[index]):
            indexs.append(index)

        if len(indexs) == 2:
          if self.cards[indexs[0]] == self.previousCards[indexs[1]] and self.cards[indexs[1]] == self.previousCards[indexs[0]]:
            if np.array_equal(self.previousCardsState[index], self.cardsState[index]):
              return True

    elif action == "APAlibi":
      randomIndex = random.randint(0, len(self.alibiCardsDict) - 1)
      randomAlibiCard = self.alibiCardsDict.pop(randomIndex)
      if randomAlibiCard:
        self.addJackHourglasses(randomAlibiCard[1])
        return True

    return False
  
  def printState(self):
    cards_state = ""

    for i in range(9):
      if i%3 == 0:
        cards_state += '\n'
      cards_state += str(self.cards[i]) + "|"
    print(cards_state)
    with open(self.board_file, 'w') as file:
      file.write(cards_state)

    print(self.detective_pawns)
    print(self.action_pawns)
   
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

  def manhunt(self):
    self.stage = "Manhunt"

    if self.turnCount % 2 == 0: 
      self.currentPlayer = "Jack"
      print("Flip back the tokens.")
    else:
      self.currentPlayer = "Detectives"
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
    
    self.checkVictory("Appeal for Witnesses", True)

