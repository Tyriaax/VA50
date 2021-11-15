import cv2
import os
import numpy as np
from numpy.lib.type_check import imag
from pynput.mouse import Listener

class Board():
  def __init__(self) -> None:
      self.cards = [0,0,0,0,0,0,0,0,0]
      self.detective_pawns = [0,0,0]
      self.board_file = os.path.abspath(os.path.join(os.path.dirname( __file__ ),"Game_state","JackPocketBoard.txt"))
  
  def printBoard(self):
    cards_state = ""

    for i in range(9): 
      if i%3 == 0:
        cards_state += '\n'
      cards_state += self.cards[i] + "|"

    with open(self.board_file, 'w') as file:
      file.write(cards_state)
  
  def getBoard(self):
    return self.cards
  def getDetectivePawn(self):
    return self.detective_pawns

"""z = [["blue","brown", "white"],["purple","black","rose"],["orange", "yellow","green"]]
board = Board(z)
board.printBoard()"""

