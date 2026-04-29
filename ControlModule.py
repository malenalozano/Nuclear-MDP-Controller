# Import required dependencies
import numpy as np
import mdptoolbox

class ControlModule:
    def __init__(self):
        """ Dummy constructor to use the Python Class as a namespace """
        pass

    @staticmethod
    def generate_P(probs: np.ndarray, n_states: np.int32, n_actions: np.int32) -> np.ndarray:
        """ Function that generates the probabilities (transition) matrix """
        ### TO BE COMPLETED BY THE STUDENTS ###
        # Matriz de transición: (n_actions x n_states x n_states)
        P = np.zeros((n_actions, n_states, n_states))

        # Desplazamientos posibles para cada acción: decrease, maintain, increase
        deltas = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]

        for a in range(n_actions):
            for s in range(n_states):
                for i, delta in enumerate(deltas[a]):
                # Estado destino con clip para no salirse de los bordes
                    s_next = int(np.clip(s + delta, 0, n_states - 1))
                    P[a][s][s_next] += probs[a][i]

        return P

    @staticmethod
    def generate_R(demand: np.float64, n_states: np.int32, n_actions: np.int32) -> np.ndarray:
        """ Function that generates the rewards (costs) matrix """
        ### TO BE COMPLETED BY THE STUDENTS ###
        # Matriz de costes: (n_actions x n_states x n_states)
        R = np.zeros((n_actions, n_states, n_states))

        for a in range(n_actions):
            for s in range(n_states):
                for s_next in range(n_states):
                    # Potencia normalizada del estado destino
                    nivel = s_next / 100.0
                    distancia = abs(demand - nivel)

                    # Penalizar x2 si la acción aleja del objetivo
                    if a == 2 and nivel > demand:    # increase alejándose por arriba
                        R[a][s][s_next] = 2 * distancia
                    elif a == 0 and nivel < demand:  # decrease alejándose por abajo
                        R[a][s][s_next] = 2 * distancia
                    else:
                        R[a][s][s_next] = distancia

        return R

    @staticmethod
    def control_iteration(demand: np.float64, current_state: np.int32, P: np.ndarray, n_states: np.int32, n_actions: np.int32, gamma: np.float64) -> np.int32:
        """ Function that computes one control-iteration """
        ### TO BE COMPLETED BY THE STUDENTS ###
        # Generar la matriz de costes para la demanda actual
        R = ControlModule.generate_R(demand, n_states, n_actions)

        # Crear y resolver el MDP con Value Iteration
        mdp = mdptoolbox.mdp.ValueIteration(P, R, gamma)
        mdp.run()

        # Devolver la acción óptima para el estado actual
        return mdp.policy[current_state]

    @staticmethod
    def control_loop(demand: np.ndarray, 
                     probs: np.ndarray,
                     n_states: np.int32, 
                     n_actions: np.int32,
                     gamma: np.float64) -> np.ndarray:
        """ Function that computes all the required iterations (control-loop) to satisfy the power demand """
        # Calcular P una sola vez (se mantiene constante)
        P = ControlModule.generate_P(probs, n_states, n_actions)

        # Inicializar el estado más cercano al primer punto de demanda
        current_state = int(np.clip(demand[0] * 100, 0, n_states - 1))

        # Array de respuesta
        response = np.zeros(len(demand), dtype=np.float64)

        for t in range(len(demand)):
            # Obtener la acción óptima para este instante
            action = ControlModule.control_iteration(demand[t], current_state, P, n_states, n_actions, gamma)

            # Simular la transición estocástica según las probabilidades del reactor
            deltas = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
            delta = np.random.choice(deltas[action], p=probs[action])

            # Actualizar estado con clip para no salirse de los bordes
            current_state = int(np.clip(current_state + delta, 0, n_states - 1))

            # Guardar la respuesta normalizada entre 0 y 1
            response[t] = current_state / 100.0

        return response
        ### ###
