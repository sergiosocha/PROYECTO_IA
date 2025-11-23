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
                continue
            red_pieces += 1
    if red_pieces - yellow_pieces == 0:
        return -1
    return 1


class HumbleButHonest(Policy):

    def mount(self) -> None:
        self.e = 0.1
        self.q_values = {1: {}, -1: {}}
        self.last_action = None
        self.last_state = None

        json_path = os.path.join(POLICY_DIR, "qvals.json")
        if os.path.exists(json_path):
            self.q_values = json.load(open(json_path))
    
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
        
        
        adyacente = 0
        for r in range(6):
            for c in range(7):
                if board[r, c] == player:
                    adyacente += self.contar_adyacentes(board, r, c, player) * 0.01
                elif board[r, c] == opponent:
                    adyacente -= self.contar_adyacentes(board, r, c, opponent) * 0.01
        score += adyacente
        
   
        return np.tanh(score)

    def actualiza_q(self, reward: float):
     
        if self.last_action is None or self.last_state is None:
            return
            
        state_codificado = get_state_codificado(self.last_state)
        action = str(self.last_action)  

        if state_codificado not in self.q_values:
            self.q_values[state_codificado] = {}
        
        if action not in self.q_values[state_codificado]:
            self.q_values[state_codificado][action] = {"q_value": 0.0, "count": 0}
        

        old_q = self.q_values[state_codificado][action]["q_value"]
        n = self.q_values[state_codificado][action]["count"] + 1
        new_q = old_q + (1 / n) * (reward - old_q)
        
        self.q_values[state_codificado][action]["q_value"] = new_q
        self.q_values[state_codificado][action]["count"] = n
        

        try:
            json_path = os.path.join(POLICY_DIR, "qvals.json")
            with open(json_path, "w") as f:
                json.dump(self.q_values, f, indent=2)
        except:
            pass


    def ult_jugada(self, board: np.ndarray, col: int):
        for r in reversed(range(6)):
            if board[r, col] != 0:
                return r, col
        return None, None
    
    def calcular_reward_incremental(self, estado_anterior: ConnectState, estado_nuevo: ConnectState, accion: int, resultado_final: str = None) -> float:
        
        if resultado_final == "win":
            return 1.0
        elif resultado_final == "lose":
            return -1.0
        elif resultado_final == "block":
            return 0.7
        
        
        player = estado_nuevo.player
        
       
        score_anterior = self.evaluar_estado(estado_anterior.board, player)
        score_nuevo = self.evaluar_estado(estado_nuevo.board, player)
        
       
        mejora = score_nuevo - score_anterior
        
       
        opponent = -player
        
        
        tres_antes = self.contar_tres_abiertos(estado_anterior.board, player)
        tres_despues = self.contar_tres_abiertos(estado_nuevo.board, player)
        if tres_despues > tres_antes:
            mejora += 0.2
        
        
        tres_rival_antes = self.contar_tres_abiertos(estado_anterior.board, opponent)
        tres_rival_despues = self.contar_tres_abiertos(estado_nuevo.board, opponent)
        if tres_rival_despues < tres_rival_antes:
            mejora += 0.15
        
       
        if not estado_nuevo.is_final():
            peligro_creado = False
            for col in range(7):
                if estado_nuevo.is_applicable(col):
                    sim_state = self.safe_transition(estado_nuevo, col)
                    if sim_state and sim_state.get_winner() == opponent:
                        peligro_creado = True
                        break
            if peligro_creado:
                mejora -= 0.3
        
     
        return np.clip(mejora, -1.0, 1.0)


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
            return

    # TODO: Revisar cuando realmente hay que actualizar_q, debería ser DESPUÉS de tomar la acción y viendo el resultdo, no antes de tomar la acción
    # Sugiero calcular de alguna forma un reward comparando self.last_state con el state que recibimos?
    def act(self, s: np.ndarray) -> int:
         
        if self.last_state is not None and self.last_action is not None:
            estado_anterior = self.last_state
            estado_actual = ConnectState(board=s, player=-self.last_state.player)
            reward = self.calcular_reward_incremental(
                estado_anterior, 
                estado_actual,
                self.last_action
            )
            self.actualiza_q(reward)
    
        player = identificar_jugador(s)
        state = ConnectState(board=s, player=player)
        state_codificado = get_state_codificado(s=state)
        opponent = -player
        cols_disponibles = [c for c in range(7) if state.is_applicable(c)]
        self.last_state = state

        if not cols_disponibles:
            self.last_state = None
            self.last_action = None
            return 0
        
        if np.all(s == 0):  # Primer movimiento del juego
            self.last_action = 3
            self.last_state = state
            return 3
        
        # REvisar si podemos ganar 
        for col in cols_disponibles:
            new_state = self.safe_transition(state, col)
            if new_state and new_state.get_winner() == player:
                self.last_action = col
                self.last_state = state
                return col
        

        # Aplicar estrategia yoConfio para tratar de bloquear al oponente si va a ganar
        for col in cols_disponibles:
            opponent_state = ConnectState(board=s.copy(), player=opponent)
            opp_new_state = self.safe_transition(opponent_state, col)
            if opp_new_state and opp_new_state.get_winner() == opponent:
                self.last_action = col
                self.last_state = state
                return col
        
        # Si e (epsilon) es mayor que el valor random, exploramos con una columna random
        # Valdría la pena cualquier columna disponible? o una random dentro de las que no tenemos guardadas en el JSON? (REVISAR)
        if self.e > np.random.rand():
            
            rng = np.random.default_rng()
            best_col = int(rng.choice(cols_disponibles))
            self.last_action = best_col
            self.last_state = state
            return best_col
        

        scores = {c: 0.0 for c in cols_disponibles}

        if state_codificado in self.q_values:
            tiene_qvalues = False
            for col in cols_disponibles:
                col_str = str(col)
                if col_str in self.q_values[state_codificado]:
                    q_val = self.q_values[state_codificado][col_str]["q_value"]
                    scores[col] = q_val * 100.0
                    tiene_qvalues = True

            if tiene_qvalues:
                for col in cols_disponibles:
                    new_state = self.safe_transition(state, col)
                    if new_state is None:
                        continue
                    estado_score = self.evaluar_estado(new_state.board, player)
                    scores[col] += estado_score * 10.0 
        else: 

            for col in cols_disponibles:
                new_state = self.safe_transition(state, col)
                if new_state is None:
                    continue
                
                estado_score = self.evaluar_estado(new_state.board, player)
                scores[col] += estado_score * 50.0

        for col in cols_disponibles:
            new_state = self.safe_transition(state, col)
            if new_state is None or new_state.is_final():
                continue
            
            for col_op in range(7):
                if not new_state.is_applicable(col_op):
                    continue
                new_op_state = self.safe_transition(new_state, col_op)
                if new_op_state and new_op_state.get_winner() == opponent:
                    scores[col] -= 500.0
                    break
        
        if 3 in cols_disponibles:
            scores[3] += 5.0

        best_col = max(scores, key=scores.get)

        self.last_action = best_col
        self.last_state = state

        return best_col

        # # En cambio si e es menor, pero ya estuvimos en este estado antes, escogemos la acción con la que mejor nos fue
        # elif state_codificado in self.q_values[player]:
        #     best_action_value   = -1000
        #     worst_action_value  = 1000
        #     for action in cols_disponibles:
        #         action_value = self.q_values[player][state_codificado].get(action, {}).get("q_value", 0.0)
        #         if action_value > best_action_value:
        #             best_action_value = action_value
        #             best_col = action
        #         elif action_value < worst_action_value:
        #             worst_action_value = action_value
        #     # Si todas las acciones guardadas tienen el mismo valor, nos vamos con cualquiera, da igual
        #     if best_action_value == worst_action_value:
        #         best_col = np.random.default_rng().choice(cols_disponibles)
        #     self.last_action = best_col
        #     return best_col

        # # Si absolutamente nunca habíamos estado en este estado continuar con la estrategia de siempre
        # # TODO: Toca ajustarla a los últimos cambios xd
        # prioridades = {
        #     "ganar": 5,
        #     "bloquear": 1.5,
        #     "tres_mios": 0.9,
        #     "tres_rival": 0.7,
        #     "antimoricion": 0.6,
        #     "adyacente": 0.02,
        #     "centro": 0.01,
        # }
        # scores = {c: 0 for c in cols_disponibles}
        # for col in cols_disponibles:
        #     new_state = self.safe_transition(state, col)
        #     if new_state and new_state.get_winner() == player:
        #         scores[col] += prioridades["ganar"]
        # for col in cols_disponibles:
        #     new_state = self.safe_transition(state, col)
        #     if new_state and new_state.get_winner() == opponent:
        #         scores[col] += prioridades["bloquear"]
        # for col in cols_disponibles:
        #     sim_state = self.safe_transition(state, col)
        #     if sim_state is None:
        #         continue
        #     if sim_state.is_final():
        #         continue
        #     for col_op in range(7):
        #         if not sim_state.is_applicable(col_op):
        #             continue
        #         new_op_state = self.safe_transition(sim_state, col_op)
        #         if new_op_state and new_op_state.get_winner() == opponent:
        #             scores[col] -= prioridades["antimoricion"]
        # for col in cols_disponibles:
        #     new_state = self.safe_transition(state, col)
        #     if new_state is None:
        #         continue
        #     tres_mios = self.contar_tres_abiertos(new_state.board, player)
        #     tres_rival = self.contar_tres_abiertos(new_state.board, opponent)
        #     if tres_mios > 0:
        #         scores[col] += prioridades["tres_mios"]
        #     if tres_rival > 0:
        #         scores[col] += prioridades["tres_rival"]
        # for col in cols_disponibles:
        #     new_state = self.safe_transition(state, col)
        #     if new_state is None:
        #         continue
        #     row, _ = self.ult_jugada(new_state.board, col)
        #     if row is not None:
        #         ady = self.contar_adyacentes(new_state.board, row, col, player)
        #         scores[col] += ady * prioridades["adyacente"]
        # if 3 in cols_disponibles:
        #     scores[3] += prioridades["centro"]
        # for col in cols_disponibles:
        #     key = get_state_codificado(state, col)
        #     scores[col] += self.q_vals.get(key, 0)
        # best_col = max(scores, key=scores.get)
        # self.last_action = best_col
        # self.actualiza_q(scores[best_col] / 1000000)
        # return best_col
