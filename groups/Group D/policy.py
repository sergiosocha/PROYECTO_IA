import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState

class PicasPolicy(Policy):

    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        state = ConnectState(s)
        player = state.player

        valid_moves = []
        for c in range(7):
            if state.is_applicable(c):
                valid_moves.append(c)

        if len(valid_moves) == 0:
            return 0

        for col in valid_moves:
            new_state = state.transition(col)
            if new_state.get_winner() == player:
                return col

        opponent = -player
        for col in valid_moves:
            new_state = state.transition(col)
            if new_state.get_winner() == opponent:
                return col

        if 3 in valid_moves:
            return 3

        return valid_moves[0]
