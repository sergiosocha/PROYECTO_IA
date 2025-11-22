import numpy as np
import os
import json
from connect4.policy import Policy
from connect4.connect_state import ConnectState


POLICY_DIR = os.path.dirname(os.path.abspath(__file__))

def __init__(self):
    self.e = 0.1
    self.q_vals = {}
    self.q_counts = {}
    self.last_action = None
    self.last_state = None


def get_qval_codificado(s: ConnectState, a: int) -> str:
    player = "Y" if s.player == 1 else "R"
    state_codificado = s.board.tobytes().hex()
    return f"{player}_state_{state_codificado}_action_{a}"

def identificar_jugador(s: np.ndarray):
    yellow_pieces = 0
    red_pieces = 0
    for r in range(6):
        for c in range(7):
            if s[r, c] == 0:
                continue
            if s[r, c] == 1:
                yellow_pieces += 1
                continue
            red_pieces += 1
    if red_pieces - yellow_pieces == 0:
        return -1
    return 1

class HumbleButHonest(Policy):

    def mount(self) -> None:
        self.e = 0.1
        self.q_vals = {}
        self.q_counts = {}
        self.last_action = None
        self.last_state = None
        json_path = os.path.join(POLICY_DIR, "qvals.json")
        if os.path.exists(json_path):
            data = json.load(open(json_path))
            self.q_vals = data.get("vals", {})
            self.q_counts = data.get("counts", {})


    def actualiza_q(self, reward: float):
        if self.last_action is not None and self.last_state is not None:
            s = self.last_state
            a = self.last_action
            key = get_qval_codificado(s, a)
            if key in self.q_counts:
                self.q_counts[key] += 1
            else:
                self.q_counts[key] = 1
            n = self.q_counts[key]
            if key not in self.q_vals:
                self.q_vals[key] = 0.0
            old_q = self.q_vals[key]
            new_q = old_q + (1 / n) * (reward - old_q)
            self.q_vals[key] = new_q
            try:
                json_path = os.path.join(POLICY_DIR, "qvals.json")
                with open(json_path, "w") as f:
                    json.dump(
                        {"vals": self.q_vals, "counts": self.q_counts},
                        f,
                        indent=4
                    )
            except:
                pass

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

    def act(self, s: np.ndarray) -> int:
        player = identificar_jugador(s)
        state = ConnectState(board=s, player=player)
        opponent = -player
        cols_disponibles = [c for c in range(7) if state.is_applicable(c)]
        self.last_state = state
        if not cols_disponibles:
            return 0
        if np.all(s == 0):
            self.last_action = 3
            return 3
        if self.e > np.random.rand():
            rng = np.random.default_rng()
            best_col = int(rng.choice(cols_disponibles))
            self.last_action = best_col
            self.actualiza_q(0)
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
            key = get_qval_codificado(state, col)
            scores[col] += self.q_vals.get(key, 0)
        best_col = max(scores, key=scores.get)
        self.last_action = best_col
        self.actualiza_q(scores[best_col] / 1000000)
        return best_col
