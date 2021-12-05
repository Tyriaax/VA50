import cv2
import os
from enum import Enum

class GameStates(Enum):
  GSWaitingHomography = 1
  GSWaitingFirstRecognition = 2
  GSGameStarted = 3
  """
  GSWaitThrowActionPawns = 4
  GSReturnActionsPawns = 5
  """

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
  
  def getCards(self):
    return self.cards

  def getPreviousCards(self):
    return self.previousCards

  def getAlibiCardsDict(self):
    return self.alibiCardsDict
  
  def setAlibiCardsDict(self, alibiCardDict):
    self.alibiCardsDict = alibiCardDict

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

  def setCards(self,cards):
    self.previousCards = self.cards
    self.cards = cards

  def setDetectivePawns(self,detective_pawns):
    self.previousDetectivePawns = self.detective_pawns
    self.detective_pawns = detective_pawns

  def getActionPawns(self):
    return self.action_pawns

  def setActionPawns(self, action_pawns):
    self.action_pawns = action_pawns

  def updateGameStatus(self):
    if self.state.value < len(GameStates):
      self.state = GameStates(self.state.value + 1)

  def getGameStatus(self):
    return self.state

class JackPocketGame():
  def __init__(self) -> None:
      self.turnCount = 1
      self.maxTurnCount = 8 
      self.numberOfSuspects = 9
      self.jackHourglasses = 0
      self.gameStage = ["Manhunt", "Appeal for Witnesses"]
      self.jackWins = False
      self.detectiveWins = False
      self.turn = ""
      #self.cardRecogntionHelper = CardsRecognitionHelper()
  
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

  def getDetectiveWins(self):
    return self.detectiveWins
  
  def checkPiecesBorder(self): #Affiche les box pour checker si tout est à sa place et peut être détecter trnql
    pass
    #print("Boxes")
    #show Tkinter window to ask confirm
  
  def checkVictory(self, stage, isJackSeen):
    if stage == "Appeal for Witnesses":
      if self.numberOfSuspects == 1 and self.jackHourglasses == 6:
        if isJackSeen:
          self.detectiveWins = True
        #else:
          self.jackWins = True

      elif self.numberOfSuspects == 1 and self.jackHourglasses < 6:
        self.detectiveWins = True
      elif self.numberOfSuspects > 1 and self.jackHourglasses == 6:
        self.jackWins = True

  def manhunt(self):
    if self.turnCount % 2 == 0: 
      self.turn = "Jack"
      print("Flip back the tokens.")
    else:
      self.turn = "Detectives"
      print("Detective starts: you can throw the tokens")
  
  def addJackHourglasses(self, numberOfHourglasses):
    self.jackHourglasses += numberOfHourglasses

  def appealOfWitnesses(self, isJackSeen):
    if isJackSeen:
      print("Turn all the cards that are not in sight to their empty side")
      print("Detective wins this turn and get the round token !")
    else:
      self.jackHourglasses += 1
      print("Turn all the cards in sight to their empty side")
      print("Jack wins this turn and get the round token !")
    
    self.checkVictory("Appeal for Witnesses", True)

  def launchGame(self):
    while not self.jackWins and not self.detectiveWins:
      pass

    """
    
    Purpose:
      Investigator aim: Discover which identity is being used by Mr. Jack from among the nine suspects.
      To do so, there can only be one remaining Suspect before the end of the eighth turn.

      Mr.Jack : Keep his identity a secret by causing the Investigator to waste as much time as possible.
      This requires Mr. Jack to acquire six hourglasses before being indentified
    
    Turn :
      Odd Turns (1-3-5-7) -> Investigator starts
        He throws the four Action tokens in the air in such a way that they land on the table near
        the district. The faces of the tokens show which actions are available during this turn.
        Note: During the turn, be careful to leave the tokens as they were thrown, as the other sides
        will be used in the following turn.
        He selects one of the four actions and carries it out.
        Then, Mr. Jack chooses two actions from the remaining three and plays them.
        The order in which this is done is not important.
        Finally, the Investigator plays the remaining available action. Play now moves to the second
        stage, the Appeal for Witnesses.

      Even Turns (2-4-6-8) – Mr. Jack starts
        He turns over the four tokens to reveal the four actions that were hidden during
        the previous turn.
        He selects one of the four actions and carries it out.
        Then the Investigator chooses two of the remaining three actions and carries them out.
        The order in which this is done is not important.
        Finally, Mr. Jack plays the remaining available action. Play now moves to the second stage,
        the Appeal for Witnesses. 

      Manhunt :
      Appeal for Witnesses:
        Mr. Jack tells the Investigator if his character is or is not in the Detectives’ line of sight
    """

