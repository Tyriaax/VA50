import numpy as np
import copy


class JackAi():
  def jack(self, game_board): #obs : infos utiles (nb de sablier etc)
    steps = 4
    self.valid_actions = ["APJoker", "APSherlock", "APToby", "APWatson", "APReturn", "APReturn2", "APChangeCard"]
    for action in self.valid_actions:
      print(action, self.score_move(game_board, action, steps), sep = '', end='\n')
    #scores = dict(zip(valid_actions, [self.score_move(game_board, action, steps) for action in #valid_actions]))
    #return max(scores)

  def score_move(self, game_board, action, steps): #Calcul score lorsque Jack fait une action
    scoreMax = 0
    best_move = None
    next_game_boards = self.get_possible_actions(game_board, action)
    for next_game_board in next_game_boards[0]:
      #pass
      score = self.minimax(next_game_board, steps - 1, False)
      if score > scoreMax:
        best_move = next_game_boards[1]

    return best_move

  def get_possible_actions(self, game_board, action): #Effectue une action qui va cahnger le board
    copy_game_board = copy.deepcopy(game_board)
    next_game_boards = []

    if action in ["APJoker", "APSherlock", "APToby", "APWatson"]:
      actions_pawns = ["APWatson", "APSherlock", "APToby"]
      detectives_pawns = ["DPWatson", "DPSherlock", "DPToby"]
      if action == "APJoker":
        for detective_pawn in detectives_pawns:
          for move in range(2):
            next_game_boards.append((self.do_action_on_detective_pawns(game_board, detective_pawn, move), (detective_pawn, move)))
      else:
        detective_pawn = detectives_pawns[actions_pawns.index(action)]
        for move in range(1,3):
          next_game_boards.append((self.do_action_on_detective_pawns(game_board, detective_pawn, move), (detective_pawn, move)))
    
    elif action in ["APReturn", "APReturn2"]: #[index, "orientation"]
      orientations = ["Up", "Down", "Left", "Right"]
      for index in range(9):
        for orientation in orientations:
          next_game_boards.append((self.do_return_action(game_board, index, orientation), (index, orientation)))

    elif action == "APChangeCard":
      for index1 in range(9):
        for index2 in [index for index in range(9) if index != index1]:
          next_game_boards.append((self.do_change_card_action(game_board, index1, index2), (index1, index2)))

    elif action == "APAlibi":
      pass
    
    self.valid_actions.remove(action)
    return next_game_boards

  def do_return_action(self, game_board, index, orientation):
    next_game_board = game_board
    next_game_board["cardsOrientation"][index][0] = orientation

    return next_game_board
  
  def do_action_on_detective_pawns(self, game_board, detective_pawn, move_of):
    #Handle same char on same place
    next_game_board = game_board
    index_detective = next_game_board["dectectivePawns"].index(detective_pawn)
    next_game_board["dectectivePawns"][index_detective] = 0
    next_game_board["dectectivePawns"][(index_detective + move_of)%len(next_game_board["dectectivePawns"])] = detective_pawn

    return next_game_board
  
  def do_change_card_action(self, game_board, index1 , index2):
    next_game_board = game_board

    next_game_board["cardsPosition"][index1], next_game_board["cardsPosition"][index2] = next_game_board["cardsPosition"][index2], next_game_board["cardsPosition"][index1]

    next_game_board["cardsOrientation"][index1], next_game_board["cardsOrientation"][index2] = next_game_board["cardsOrientation"][index2], next_game_board["cardsOrientation"][index1]

    return next_game_board

  def do_alibi_action(self, obs):
    pass
    
  def get_heuristic(self, game_board): #Calcul la valeur heuristique pour un game_board
    correspondingIndexes = ((0,1), (0,2), (0,3), (1,4), (2,4), (3,4), (4,3), (4,2), (4,1), (3,0), (2,0), (1,0))

    jack_index = game_board["cardsPosition"].index(game_board["jack"])
    jack_index_x, jack_index_y = jack_index//3 + 1 , jack_index%3 + 1
    detectivesPosition = [] 

    matrix = np.zeros((5, 5), dtype= np.chararray)
    for index in range(len(game_board["dectectivePawns"])):
      if game_board["dectectivePawns"][index] in ["DPWatson", "DPSherlock", "DPToby"]:
        detectivesPosition.append([correspondingIndexes[index][0], correspondingIndexes[index][1]])
        matrix[correspondingIndexes[index][0]][correspondingIndexes[index][1]] = game_board["dectectivePawns"][index]

    for index in range(len(game_board["cardsOrientation"])):
      matrix[index//3 + 1][index%3 + 1] = game_board["cardsOrientation"][index]
    
    print(matrix)

    #ligne
    jack_in_sight = False
    for detectivePosition in detectivesPosition:
      if detectivePosition[0] == jack_index_x: #check ligne
        if detectivePosition[1] == 0:
          for index in range(1,4):
            if matrix[detectivePosition[0]][index][0] in ["Left", "Up", "Down"]:
              if jack_index_y == index:
                jack_in_sight = True
        elif detectivePosition[1] == 4:
          for index in range(1,4):
            if matrix[detectivePosition[0]][index][0] in ["Right", "Up", "Down"]:
              if jack_index_y == index:
                jack_in_sight = True
      elif detectivePosition[1] == jack_index_y: #check colonne
        if detectivePosition[0] == 0:
          for index in range(1,4):
              if matrix[index][detectivePosition[1]][0] in ["Right", "Up", "Left"]:
                if jack_index_x == index:
                  jack_in_sight = True
        elif detectivePosition[0] == 4:
          if matrix[index][detectivePosition[1]][0] in ["Right", "Down", "Left"]:
              if jack_index_x == index:
                  jack_in_sight = True
    
    if jack_in_sight:
      return 10000
    else:
      return -1000

  def is_terminal_node(self, game_board):# check si la partie est terminé
    pass

  def minimax(self, node, depth, maximizingPlayer):
    is_terminal = self.is_terminal_node(node)
    valid_actions = "action" #get Valid moves
    if depth == 0 or is_terminal:
        return self.get_heuristic(node)
    if maximizingPlayer:
        value = -np.Inf
        for action in self.valid_actions:
          childs = self.get_possible_actions(node, action)
          for child in childs[0]:
            value = max(value, self.minimax(child, depth-1, False))
        return value
    else:
        value = np.Inf
        for action in self.valid_actions:
          childs = self.get_possible_actions(node, action)
          for child in childs[0]:
            value = min(value, self.minimax(child, depth-1, True))
        return value


game_board = {
  "cardsPosition" : ["red", "blue", "black", "purple", "pink", "yellow", "brown", "orange", "white"],
  "cardsOrientation" : [["Left", "front"], ["Right", "front"], ["Down", "front"], ["Up", "front"], ["Left", "front"], ["Left", "front"], ["Down", "front"], ["Up", "front"], ["Up", "front"]],
  "dectectivePawns" : ["DPWatson", 0, 0, 0, "DPToby", 0,0,0, "DPSherlock",0,0,0],
  "hourglasses" : 4,
  "jack" : "purple" 
} 

#Problème référence dans get_possible_actions
#Ajout de plusieurs dp sur une case
#Action alibi

#pop action faite

a = JackAi()
#a.do_return_action(game_board,0, "left")
a.jack(game_board)
#score = a.get_heuristic(game_board)
#print(score)


