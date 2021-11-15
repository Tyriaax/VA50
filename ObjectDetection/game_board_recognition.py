import cv2
import os
import numpy as np
from numpy.lib.type_check import imag
from pynput.mouse import Listener

class Board():
  def __init__(self, cards : list) -> None:
      self.cards = cards
      self.detective_pawns = []
      self.board_file = os.path.abspath(os.path.join(os.path.dirname( __file__ ),"Game_state","JackPocketBoard.txt"))
  
  def printBoard(self):
    cards_state = ""
    for i in range(3):
      cards_state += "\n-------------------------\n"
      for j in range(3):
        cards_state += str(self.cards[i][j]) + " | "
    
    with open(self.board_file, 'w') as file:
      file.write(cards_state)
    print(cards_state)

z = [["blue","brown", "white"],["purple","black","rose"],["orange", "yellow","green"]]
board = Board(z)
board.printBoard()

