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
  
  def getCards(self):
    return self.cards
  def getDetectivePawns(self):
    return self.detective_pawns

"""z = [["blue","brown", "white"],["purple","black","rose"],["orange", "yellow","green"]]
board = Board(z)
board.printBoard()"""

