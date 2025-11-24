# trainer.py
import os
import numpy as np

from connect4.connect_state import ConnectState
from connect4.policy import Policy
from connect4.utils import find_importable_classes


def get_humble_class():
    participants = find_importable_classes("groups", Policy)
    print("Policies encontradas:", list(participants.keys()))

    if "Group F" in participants:
        return participants["Group F"]

    # Fallback: buscar por nombre de la clase
    for name, cls in participants.items():
        if getattr(cls, "__name__", "") == "HumbleButHonest":
            return cls

    raise RuntimeError("No encontré la clase HumbleButHonest en groups/")


def jugar_partida(policy_cls):
    rojo = policy_cls()
    amarillo = policy_cls()

    rojo.mount()
    amarillo.mount()

    state = ConnectState()

    while not state.is_final():
        if state.player == -1:
            action = rojo.act(state.board)
        else:
            action = amarillo.act(state.board)

        state = state.transition(int(action))

    return state.get_winner()


def entrenar(episodios: int = 200):
    policy_cls = get_humble_class()

    wins_rojo = 0
    wins_amarillo = 0
    draws = 0

    for i in range(episodios):
        resultado = jugar_partida(policy_cls)

        if resultado == -1:
            wins_rojo += 1
        elif resultado == 1:
            wins_amarillo += 1
        else:
            draws += 1

        if (i + 1) % 200 == 0:
            print(f"{i+1} partidas completadas...")

    print("\n--- RESULTADOS ENTRENAMIENTO ---")
    print("Victorias como rojo:", wins_rojo)
    print("Victorias como amarillo:", wins_amarillo)
    print("Empates:", draws)
    print("--------------------------------")


if __name__ == "__main__":
    entrenar(200)
