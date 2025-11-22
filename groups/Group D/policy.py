import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState

class PicasPolicy(Policy):

    def mount(self) -> None:
        self.qval = {c: 0.0 for c in range(7)}
        self.q_counts = {c: 0 for c in range(7)}
        self.last_action = None

    def iniciar_qval(self) -> None:
        self.qval = {c: 0.0 for c in range(7)}
        self.q_counts = {c: 0 for c in range(7)}
        self.last_action = None

    def actualiza_q(self, reward: float):
        if self.last_action is not None:
            a = self.last_action
            self.q_counts[a] += 1
            n = self.q_counts[a]
            old_q = self.qval[a]
            new_q = old_q + (1 / n) * (reward - old_q)
            self.qval[a] = new_q

    def ult_jugada(self, board: np.ndarray, col: int):
        for r in reversed(range(6)):
            if board[r, col] != 0:
                return r, col
        return None, None

    def contar_adyacentes(self, board: np.ndarray, row: int, col: int, player: int) -> int:
        ady = [
            (row, col-1), (row, col+1),
            (row+1, col), (row-1, col),
            (row-1, col-1), (row+1, col+1),
            (row-1, col+1), (row+1, col-1),
        ]
        puntos = 0
        for (r, c) in ady:
            if 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                puntos += 1
        return puntos

    def contar_tres_abiertos(self, board: np.ndarray, player: int) -> int:
        cont = 0
        for r in range(6):
            for c in range(4):
                ventana = board[r, c:c+4]
                if np.count_nonzero(ventana == player) == 3 and np.count_nonzero(ventana == 0) == 1:
                    cont += 1
        for r in range(3):
            for c in range(7):
                ventana = board[r:r+4, c]
                if np.count_nonzero(ventana == player) == 3 and np.count_nonzero(ventana == 0) == 1:
                    cont += 1
        for r in range(3):
            for c in range(4):
                ventana = np.array([board[r+i, c+i] for i in range(4)])
                if np.count_nonzero(ventana == player) == 3 and np.count_nonzero(ventana == 0) == 1:
                    cont += 1
        for r in range(3, 6):
            for c in range(4):
                ventana = np.array([board[r-i, c+i] for i in range(4)])
                if np.count_nonzero(ventana == player) == 3 and np.count_nonzero(ventana == 0) == 1:
                    cont += 1
        return cont

    def safe_transition(self, state, col):
        try:
            return state.transition(col)
        except:
            return None

    def act(self, s: np.ndarray, e: float = 0.1) -> int:
        state = ConnectState(s)
        player = state.player
        opponent = -player
        cols_disponibles = [c for c in range(7) if state.is_applicable(c)]

        if not cols_disponibles:
            return 0
        if np.all(s == 0):
            self.last_action = 3
            return 3

        if e > np.random.rand():
            rng = np.random.default_rng()
            best_col = int(rng.choice(cols_disponibles))
            self.last_action = best_col
            return best_col

        prioridades = {
            "ganar": 5000000,
            "bloquear": 150000,
            "tres_mios": 9000,
            "tres_rival": 7000,
            "antimoricion": 60000,
            "adyacente": 20,
            "centro": 10,
        }

        scores = {c: 0 for c in cols_disponibles}

        for col in cols_disponibles:
            new_state = self.safe_transition(state, col)
            if new_state and new_state.get_winner() == player:
                scores[col] += prioridades["ganar"]

        for col in cols_disponibles:
            new_state = self.safe_transition(state, col)
            if new_state and new_state.get_winner() == opponent:
                scores[col] += prioridades["bloquear"]

        for col in cols_disponibles:
            sim_state = self.safe_transition(state, col)
            if sim_state is None:
                continue
            if sim_state.is_final():
                continue
            for col_op in range(7):
                if not sim_state.is_applicable(col_op):
                    continue
                new_op_state = self.safe_transition(sim_state, col_op)
                if new_op_state and new_op_state.get_winner() == opponent:
                    scores[col] -= prioridades["antimoricion"]

        for col in cols_disponibles:
            new_state = self.safe_transition(state, col)
            if new_state is None:
                continue
            tres_mios = self.contar_tres_abiertos(new_state.board, player)
            tres_rival = self.contar_tres_abiertos(new_state.board, opponent)
            if tres_mios > 0:
                scores[col] += prioridades["tres_mios"]
            if tres_rival > 0:
                scores[col] += prioridades["tres_rival"]

        for col in cols_disponibles:
            new_state = self.safe_transition(state, col)
            if new_state is None:
                continue
            row, _ = self.ult_jugada(new_state.board, col)
            if row is not None:
                ady = self.contar_adyacentes(new_state.board, row, col, player)
                scores[col] += ady * prioridades["adyacente"]

        if 3 in cols_disponibles:
            scores[3] += prioridades["centro"]


        for col in cols_disponibles:
            scores[col] += self.qval.get(col, 0)

        best_col = max(scores, key=scores.get)
        self.last_action = best_col
        return best_col
