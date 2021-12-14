import numpy as np

class JackAi():
  def jack(self, obs, game_board): #obs : infos utiles (nb de sablier etc)
    steps = 4
    valid_actions = "action"
    scores = dict(zip(valid_actions, [self.score_move(game_board, action, steps) for action in valid_actions]))
    return max(scores)

  def score_move(self, game_board, action, steps): #Calcul score lorsque Jack fait une action
    next_game_board = self.do_action(game_board, action)
    score = self.minimax(next_game_board, steps - 1, False)
    return score
  
  def do_action(self, game_board, action): #Effectue une action qui va cahnger le board
    if action in ["APJoker", "APHolmes", "APToby", "APWatson"]:
      pass
    elif action in ["APReturn", "APReturn2"]:
      pass
    elif action == "APChangeCard":
      pass
    elif action == "APAlibi":
      pass

  def get_heuristic(self, game_board): #Calcul la valeur heuristique pour un game_board
    pass
    # 100000 quand il gagne
    # 1 point quand il gagne un sablier
    # -10 quand il perd un sablier

  def is_terminal_node(self, game_board):# check si la partie est terminé
    pass

  def minimax(self, node, depth, maximizingPlayer):
    is_terminal = self.is_terminal_node(node)
    valid_actions = "action" #get Valid moves
    if depth == 0 or is_terminal:
        return self.get_heuristic(node)
    if maximizingPlayer:
        value = -np.Inf
        for action in valid_actions:
            child = self.do_action(node, action)
            value = max(value, self.minimax(child, depth-1, False))
        return value
    else:
        value = np.Inf
        for col in valid_actions:
            child = self.do_action(node, col)
            value = min(value, self.minimax(child, depth-1, True))
        return value



  
 
  

