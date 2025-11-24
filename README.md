
# Connect 4 - Agente HumbleButHonest

Creación de un agente que juegue Connect 4 y sea capaz de ganar todas las partidas, ademas sea capaz de tomar decisiones correctas
según las circunstancias del juego, es decir que sepa como reaccionar dependiendo la jugada del rival

## Grupo

Proyecto IA
- Felipe Ballesteros
- Sebastian Piñeros Castellanos
- Sergio Eduardo Socha Mendoza

## Descripción

Este proyecto implementa un agente que juega Connect 4. El agente aprende de cada partida y almacena su conocimiento en un archivo JSON (`qvals.json`), mejorando continuamente su desempeño.

### Características Principales

- **Q-Learning clásico** con actualización incremental
- **Epsilon-greedy** para balance exploración/explotación
- **Persistencia de conocimiento** en `qvals.json`

## Instalación

### Requisitos

- Python 3.13 o superior (Recomendado)
- numpy
- matplotlib
- pydantic

### Pasos de instalación

```bash
# Clonar el repositorio
git clone https://github.com/sergiosocha/PROYECTO_IA.git
cd PROYECTO_IA

# Instalar dependencias
pip install numpy matplotlib pydantic
```

## Uso

### Opción 1: Torneo suministrado por el docente

Si utilizas el entorno de torneo proporcionado por el docente:

```bash
# Ejecutar el torneo
python main.py
```

El agente se encuentra en `groups/Group F/policy.py` y participará automáticamente en el torneo contra otros agentes.

### Opción 2: Uso en gradescope

Subir el agente a gradescope junto con su respectivo JSON (`qvals.json`)

### Requisitos del Agente

Para que el agente sea válido en el torneo:
- Debe heredar de `Policy` (clase base del framework)


## Archivos del Proyecto

```
PROYECTO_IA/
├── groups/
|    Group D/
│       ├── policy.py          # Agente Heuristico de entrenamiento (yoConfio.py) es el agente que se realizo en primera                        instancia dicho agente esta hecho para siempre buscar ganar o bloquear perfecto para entrenar a                              nuestra policy
│   └── Group F/
│       ├── policy.py          # Agente principal (HumbleButHonest)
│       └── qvals.json         # Conocimiento aprendido (generado automáticamente)
├── connect4/
│   ├── connect_state.py       # Lógica del juego
│   └── policy.py              # Clase base Policy
├── tournament.py              # Sistema de torneos
├── main.py                    # Punto de entrada
└── trainner.py                # Nos ayuda a generar muchos partidos de prueba para entrenar nuestro modelo
```


## Aprendizaje Continuo

El agente almacena su conocimiento en `qvals.json`. Este archivo:
- Se crea automáticamente en la primera ejecución
- Se actualiza después de cada jugada
- Persiste entre ejecuciones
- Contiene Q-values por estado y acción

**Para entrenar el agente**: ejecute el archivo trainer.py

```bash
# Ejecutar el torneo
python trainer.py
```

se ejecutaran almenos 2000 juegos para que el agente tenga datos de entrenamiento suficientes para iniciar

**Para resetear el aprendizaje**: Elimina `qvals.json` y se reiniciará desde cero.




