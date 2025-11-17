import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class YoConfio(Policy):

    def mount(self) -> None:
        pass

    def get_ult_juga(self, board: np.ndarray, col: int):
        """Devuelve (row, col) donde cayó la ficha en el tablero resultante."""
        for r in reversed(range(6)):
            if board[r, col] != 0:
                return r, col
        return None, None

    def contar_adyacentes(self, board: np.ndarray, row: int, col: int, player: int) -> int:
        """Cuenta cuántas fichas del jugador están adyacentes (8 direcciones)."""
        adyacentes = [
            (row, col-1), (row, col+1),      
            (row+1, col), (row-1, col),      
            (row-1, col-1), (row+1, col+1),  
            (row-1, col+1), (row+1, col-1),  
        ]

        puntos = 0
        for (r, c) in adyacentes:
            if 0 <= r < 6 and 0 <= c < 7:
                if board[r, c] == player:
                    puntos += 1
        return puntos

    def act(self, s: np.ndarray) -> int:
        state = ConnectState(s)

        player = state.player
        opponent = -player

        cols_disponibles = state.get_free_cols()

        if np.all(s == 0):
            return 3

        prioridades = {
            "ganar": 500,
            "bloquear": 300,
            "centro": 5,
            "adyacente": 20
        }

        scores = {col: 0 for col in cols_disponibles}

        for col in cols_disponibles:
            new_state = state.transition(col)
            if new_state.get_winner() == player:
                scores[col] += prioridades["ganar"]

        for col in cols_disponibles:
            new_state = state.transition(col)
            if new_state.get_winner() == opponent:
                scores[col] += prioridades["bloquear"]

        for col in cols_disponibles:
            new_state = state.transition(col)

            row, _ = self.get_ult_juga(new_state.board, col)
            if row is None:
                continue

            num_ady = self.contar_adyacentes(new_state.board, row, col, player)
            scores[col] += num_ady * prioridades["adyacente"]

        if 3 in cols_disponibles:
            scores[3] += prioridades["centro"]

        return max(scores, key=scores.get)
