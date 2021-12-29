import numpy as np
import copy


class JackAi():
  def jack(self, game_board, turn, is_jack_first, valid_actions): #isJackFirst = True, turn 1 else , turn = 2
    steps = 4 - turn
    best_action, best_score = None, - np.Inf
    for action in valid_actions:
      jack_step = self.score_move(game_board, action, steps, copy.deepcopy(valid_actions), turn, is_jack_first)
      if jack_step[1] > best_score:
        best_action = [action, jack_step[0][1]]

    return best_action

  def score_move(self, game_board, action, steps, valid_actions, turn, is_jack_first): #Calcul score lorsque Jack fait une action
    scoreMax = - np.Inf
    best_move = None

    next_game_boards, valid_actions_remaining = self.get_possible_actions(game_board, action, copy.deepcopy(valid_actions))
    for next_game_board in next_game_boards:
      score = self.minimax(next_game_board[0], steps - 1, is_jack_first, valid_actions_remaining, turn)
      if score > scoreMax:
        scoreMax = score
        best_move = next_game_boards[1]

    return best_move, scoreMax

  def get_possible_actions(self, game_board, action, valid_actions): #Effectue une action qui va cahnger le board
    next_game_boards = []

    if action in ["APJoker", "APSherlock", "APToby", "APWatson"]:
      actions_pawns = ["APWatson", "APSherlock", "APToby"]
      detectives_pawns = ["DPWatson", "DPSherlock", "DPToby"]
      if action == "APJoker":
        for detective_pawn in detectives_pawns:
          for move in range(2):
            next_game_boards.append((self.do_action_on_detective_pawns(copy.deepcopy(game_board), detective_pawn, move), (detective_pawn, move)))
      else:
        detective_pawn = detectives_pawns[actions_pawns.index(action)]
        for move in range(1,3):
          next_game_boards.append((self.do_action_on_detective_pawns(copy.deepcopy(game_board), detective_pawn, move), (detective_pawn, move)))
    
    elif action in ["APReturn", "APReturn2"]: #[index, "orientation"]
      orientations = ["Up", "Down", "Left", "Right"]
      for index in range(9):
        for orientation in orientations:
          next_game_boards.append((self.do_return_action(copy.deepcopy(game_board), index, orientation), (index, orientation)))

    elif action == "APChangeCard":
      for index1 in range(9):
        for index2 in [index for index in range(9) if index != index1]:
          next_game_boards.append((self.do_change_card_action(copy.deepcopy(game_board), index1, index2), (index1, index2)))

    elif action == "APAlibi":
      pass
    
    valid_actions.remove(action)
    return next_game_boards, valid_actions

  def do_return_action(self, game_board, index, orientation):
    next_game_board = game_board
    next_game_board["cardsOrientation"][index][0] = orientation

    return next_game_board
  
  def do_action_on_detective_pawns(self, game_board, detective_pawn, move_of):
    #Handle same char on same place
    next_game_board = game_board
    # for element_detective_pawns in next_game_board["dectectivePawns"]:
    #   print(element_detective_pawns)
    #   if detective_pawn in element_detective_pawns:
    
    index_detective = next_game_board["dectectivePawns"].index(detective_pawn)
    
    # if type(next_game_board["dectectivePawns"][index_detective]) == list():
    #   next_game_board["dectectivePawns"][index_detective].remove(detective_pawn)
    # else:
    #   next_game_board["dectectivePawns"][index_detective] = 0
    # destination_index = (index_detective + move_of)%len(next_game_board["dectectivePawns"])
    # if  next_game_board["dectectivePawns"][destination_index] == 0:
    #   next_game_board["dectectivePawns"][destination_index] = detective_pawn
    # else:
    #   next_game_board["dectectivePawns"][destination_index] = [next_game_board["dectectivePawns"][destination_index], detective_pawn]

    destination_index = (index_detective + move_of)%len(next_game_board["dectectivePawns"])
    if next_game_board["dectectivePawns"][destination_index] == 0:
      next_game_board["dectectivePawns"][index_detective] = 0
      next_game_board["dectectivePawns"][destination_index] = detective_pawn

    return next_game_board
  
  def do_change_card_action(self, game_board, index1 , index2):
    next_game_board = game_board

    next_game_board["cardsPosition"][index1], next_game_board["cardsPosition"][index2] = next_game_board["cardsPosition"][index2], next_game_board["cardsPosition"][index1]

    next_game_board["cardsOrientation"][index1], next_game_board["cardsOrientation"][index2] = next_game_board["cardsOrientation"][index2], next_game_board["cardsOrientation"][index1]

    return next_game_board

  def do_alibi_action(self, obs):
    pass
  
  def in_sight(self, cards, jack, orientations):
    jack_in_sight = False
    number_of_people_in_sight = 0

    for card in cards:
      if card[1][0] in orientations:
        if card[1][1] == "front":
          number_of_people_in_sight += 1   
        if jack == card[0]:
          jack_in_sight = True
      else:
        break
    
    return jack_in_sight, number_of_people_in_sight

  def get_heuristic(self, game_board):
    board_score = 0
    detectives_position = []
    number_of_people_in_sight = 0
    number_of_detectives_who_see_jack = 0
    jack_has_been_seen = False
    range_corresponding_cards = ((0, 3, 6), (1, 4, 7),(2, 5, 8),
                                (2, 1, 0), (5, 4, 3), (8, 7, 6),
                                (8, 5, 2), (7, 4, 1), (6, 3, 0),
                                (6, 7, 8), (3, 4, 5),(0, 1, 2))

    for index in range(len(game_board["dectectivePawns"])):
      if game_board["dectectivePawns"][index] in ["DPWatson", "DPSherlock", "DPToby"]:
        detectives_position.append(index)

    for detective_position in detectives_position:
      cards = []
      for index_card in range_corresponding_cards[detective_position]:
        cards.append([game_board["cardsPosition"][index_card], game_board["cardsOrientation"][index_card]]) 
      
      if detective_position in (9, 10, 11):
        jack_in_sight, in_sight_of_detective = self.in_sight(cards, game_board["jack"], ["Left", "Up", "Down"])
      elif detective_position in (3, 4, 5):
        jack_in_sight, in_sight_of_detective = self.in_sight(cards, game_board["jack"], ["Right", "Up", "Down"])
      elif detective_position in (0, 1, 2):
        jack_in_sight, in_sight_of_detective = self.in_sight(cards, game_board["jack"], ["Right", "Up", "Left"])
      elif detective_position in (6, 7, 8):
        jack_in_sight, in_sight_of_detective = self.in_sight(cards, game_board["jack"], ["Right", "Down", "Left"])

      number_of_people_in_sight += in_sight_of_detective
      if jack_in_sight:
        number_of_detectives_who_see_jack += 1
        jack_has_been_seen = True

    if jack_has_been_seen:
      return -1000 + number_of_people_in_sight * 50 - number_of_detectives_who_see_jack * 15
    else:
      return 10000 - number_of_people_in_sight * 50 

  def is_terminal_node(self, game_board):# check si la partie est terminé
    pass

  def minimax(self, node, depth, isJackFirst, valid_actions, turn):
    is_terminal = self.is_terminal_node(node)
    if depth == 0 or is_terminal:
        return self.get_heuristic(node)
    if (isJackFirst and (turn == 0 or turn == 3)) or (not isJackFirst and (turn == 1 or turn == 2)):
        value = -np.Inf
        for action in valid_actions:
          childs, valid_actions_remaining = self.get_possible_actions(node, action, copy.deepcopy(valid_actions))
          for child in childs:
            value = max(value, self.minimax(child[0], depth-1, isJackFirst, valid_actions_remaining, turn + 1))
        return value
    else:
        value = np.Inf
        for action in valid_actions:
          childs, valid_actions_remaining = self.get_possible_actions(node, action, copy.deepcopy(valid_actions))
          for child in childs:
            value = min(value, self.minimax(child[0], depth-1, isJackFirst, valid_actions_remaining, turn + 1))
        return value

def old_minimax(self, node, depth, maximizingPlayer, valid_actions):
    is_terminal = self.is_terminal_node(node)
    if depth == 0 or is_terminal:
        return self.get_heuristic(node)
    #If JackFirst and turn == 1 || 4 or not JackFirst and turn == 2 or 3:
    if maximizingPlayer:
        value = -np.Inf
        for action in valid_actions:
          childs, valid_actions_remaining = self.get_possible_actions(node, action, copy.deepcopy(valid_actions))
          for child in childs:
            value = max(value, self.minimax(child[0], depth-1, False, valid_actions_remaining))
        return value
    else:
        value = np.Inf
        for action in valid_actions:
          childs, valid_actions_remaining = self.get_possible_actions(node, action, copy.deepcopy(valid_actions))
          for child in childs:
            value = min(value, self.minimax(child[0], depth-1, True, valid_actions_remaining))
        return value


game_board = {
  "cardsPosition" : ["red", "blue", "black", "purple", "pink", "yellow", "brown", "orange", "white"],
  "cardsOrientation" : [["Left", "back"], ["Right", "front"], ["Down", "front"], ["Up", "front"], ["Left", "front"], ["Left", "front"], ["Down", "front"], ["Up", "front"], ["Up", "front"]],
  "dectectivePawns" : ["DPWatson", 0, 0, 0, "DPToby", 0,0,0, 0,0,"DPSherlock",0],
  "hourglasses" : 4,
  "jack" : "purple" 
} 

#Ajout de plusieurs dp sur une case
#Action alibi


# valid_actions = ["APJoker", "APSherlock", "APReturn"] #"APChangeCard"]

# a = JackAi()
# b = a.jack(game_board, 2, False, valid_actions)
# print(b)


