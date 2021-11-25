import cv2
import os

class GameBoard():
  def __init__(self) -> None:
      self.cards = [0,0,0,0,0,0,0,0,0]
      self.detective_pawns = [0,0,0,0,0,0,0,0,0,0,0,0]
      self.board_file = os.path.abspath(os.path.join(os.path.dirname( __file__ ),"Game_state","JackPocketBoard.txt"))
  
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
  
  def getCards(self):
    return self.cards
  def getDetectivePawns(self):
    return self.detective_pawns
  def setCards(self,cards):
    self.cards = cards
  def setDetectivePawns(self,detective_pawns):
    self.detective_pawns = detective_pawns


    """
    Turn = 0 - 8
    numberOfSuspects = 9
    hourglassesAcquired = 0 - 6
    
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

