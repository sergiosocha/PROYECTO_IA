import os
import json
import numpy as np

from connect4.connect_state import ConnectState
from connect4.policy import Policy

POLICY_DIR = os.path.dirname(os.path.abspath(__file__))


def get_state_codificado(s: ConnectState) -> str:
    return s.board.tobytes().hex()


def identificar_jugador(s: np.ndarray) -> int:
    yellow_pieces = 0
    red_pieces = 0
    for r in range(6):
        for c in range(7):
            if s[r, c] == 0:
                continue
            if s[r, c] == 1:
                yellow_pieces += 1
            else:
                red_pieces += 1
    if red_pieces - yellow_pieces == 0:
        return -1
    return 1


class HumbleButHonest(Policy):

    def mount(self, timeout: float = None) -> None:
        self.timeout = timeout
        self.e = 0.1
        self.alpha = 0.1
        self.gamma = 0.95
        self.q_values = {}
        self.last_action = None
        self.last_state = None

        json_path = os.path.join(POLICY_DIR, "qvals.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    self.q_values = json.load(f)
            except:
                pass

    def safe_transition(self, state, col):
        try:
            return state.transition(col)
        except:
            return None

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

    def evaluar_estado(self, board: np.ndarray, player: int) -> float:
        opponent = -player
        score = 0.0
        
        tres_mios = self.contar_tres_abiertos(board, player)
        tres_rival = self.contar_tres_abiertos(board, opponent)
        score += tres_mios * 0.3
        score -= tres_rival * 0.4
        
        def contar_dos_abiertos(board, p):
            cont = 0
            for r in range(6):
                for c in range(4):
                    ventana = board[r, c:c+4]
                    if np.count_nonzero(ventana == p) == 2 and np.count_nonzero(ventana == 0) == 2:
                        cont += 1
            for r in range(3):
                for c in range(7):
                    ventana = board[r:r+4, c]
                    if np.count_nonzero(ventana == p) == 2 and np.count_nonzero(ventana == 0) == 2:
                        cont += 1
            return cont
        
        dos_mios = contar_dos_abiertos(board, player)
        dos_rival = contar_dos_abiertos(board, opponent)
        score += dos_mios * 0.1
        score -= dos_rival * 0.15
        
        centro_count = 0
        for r in range(6):
            for c in [2, 3, 4]:
                if board[r, c] == player:
                    centro_count += 1
                elif board[r, c] == opponent:
                    centro_count -= 1
        score += centro_count * 0.02
        
        conectividad = 0
        for r in range(6):
            for c in range(7):
                if board[r, c] == player:
                    conectividad += self.contar_adyacentes(board, r, c, player) * 0.01
                elif board[r, c] == opponent:
                    conectividad -= self.contar_adyacentes(board, r, c, opponent) * 0.01
        score += conectividad
        
        return np.tanh(score)

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, 'e'):
            self.e = 0.1
        if not hasattr(self, 'q_values'):
            self.q_values = {}
        if not hasattr(self, 'last_action'):
            self.last_action = None
        if not hasattr(self, 'last_state'):
            self.last_state = None
        if not hasattr(self, 'alpha'):
            self.alpha = 0.1
        if not hasattr(self, 'gamma'):
            self.gamma = 0.95
        
        player = identificar_jugador(s)
        state = ConnectState(board=s, player=player)
        state_codificado = get_state_codificado(state)
        opponent = -player
        cols_disponibles = [c for c in range(7) if state.is_applicable(c)]
        
        if not cols_disponibles:
            self.last_state = None
            self.last_action = None
            return 0
        
        if np.all(s == 0):
            self.last_action = 3
            self.last_state = state
            return 3
        
        if self.last_state is not None and self.last_action is not None:
            last_state_cod = get_state_codificado(self.last_state)
            last_action_str = str(self.last_action)
            
            if last_state_cod not in self.q_values:
                self.q_values[last_state_cod] = {}
            if last_action_str not in self.q_values[last_state_cod]:
                self.q_values[last_state_cod][last_action_str] = {"q_value": 0.0, "count": 0}
            
            reward = 0.0
            if state.is_final():
                winner = state.get_winner()
                if winner == -self.last_state.player:
                    reward = 1.0
                elif winner == self.last_state.player:
                    reward = -1.0
                else:
                    reward = 0.0
            else:
                eval_antes = self.evaluar_estado(self.last_state.board, self.last_state.player)
                eval_despues = self.evaluar_estado(s, -self.last_state.player)
                reward = eval_despues - eval_antes
            
            old_q = self.q_values[last_state_cod][last_action_str]["q_value"]
            
            max_future_q = 0.0
            if state_codificado in self.q_values and not state.is_final():
                for col in cols_disponibles:
                    col_str = str(col)
                    if col_str in self.q_values[state_codificado]:
                        q = self.q_values[state_codificado][col_str]["q_value"]
                        max_future_q = max(max_future_q, q)
            
            new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
            
            self.q_values[last_state_cod][last_action_str]["q_value"] = new_q
            self.q_values[last_state_cod][last_action_str]["count"] += 1
            
            try:
                json_path = os.path.join(POLICY_DIR, "qvals.json")
                with open(json_path, "w") as f:
                    json.dump(self.q_values, f, indent=2)
            except:
                pass
        
        for col in cols_disponibles:
            new_state = self.safe_transition(state, col)
            if new_state and new_state.get_winner() == player:
                self.last_action = col
                self.last_state = state
                return col
        
        for col in cols_disponibles:
            opponent_state = ConnectState(board=s.copy(), player=opponent)
            opp_new_state = self.safe_transition(opponent_state, col)
            if opp_new_state and opp_new_state.get_winner() == opponent:
                self.last_action = col
                self.last_state = state
                return col
        
        if self.e > np.random.rand():
            best_col = int(np.random.default_rng().choice(cols_disponibles))
            self.last_action = best_col
            self.last_state = state
            return best_col
        
        if state_codificado in self.q_values:
            best_col = None
            best_q = float('-inf')
            
            for col in cols_disponibles:
                col_str = str(col)
                if col_str in self.q_values[state_codificado]:
                    q = self.q_values[state_codificado][col_str]["q_value"]
                    if q > best_q:
                        best_q = q
                        best_col = col
            
            if best_col is not None:
                self.last_action = best_col
                self.last_state = state
                return best_col
        
        best_col = int(np.random.default_rng().choice(cols_disponibles))
        self.last_action = best_col
        self.last_state = state
        return best_col