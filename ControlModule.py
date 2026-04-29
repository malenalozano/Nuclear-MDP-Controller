# Import required dependencies
import numpy as np
import mdptoolbox

# Desplazamientos asociados a cada acción del MDP, en el mismo orden que se
# almacenan las probabilidades en los ficheros JSON de los reactores:
#   - decrease (a=0): [-2, -1,  0]   (no_deseado, deseado, no_deseado)
#   - maintain (a=1): [-1,  0, +1]   (no_deseado, deseado, no_deseado)
#   - increase (a=2): [ 0, +1, +2]   (no_deseado, deseado, no_deseado)
ACTION_DELTAS = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]


class ControlModule:
    def __init__(self):
        """ Dummy constructor to use the Python Class as a namespace """
        pass

    @staticmethod
    def generate_P(probs: np.ndarray, n_states: np.int32, n_actions: np.int32) -> np.ndarray:
        """ Function that generates the probabilities (transition) matrix.

        Devuelve una matriz P de dimensiones (n_actions x n_states x n_states),
        donde P[a, s, s'] es la probabilidad de transicionar de s a s' al
        ejecutar la acción a. Las transiciones se recortan ([0, n_states - 1])
        para garantizar que la potencia no salga del rango [0, 1]; cuando
        varios desplazamientos caen en el mismo estado destino por el clipping,
        las probabilidades se acumulan, manteniendo P estocástica.
        """
        P = np.zeros((n_actions, n_states, n_states), dtype=np.float64)

        for a in range(n_actions):
            for s in range(n_states):
                for i, delta in enumerate(ACTION_DELTAS[a]):
                    s_next = int(np.clip(s + delta, 0, n_states - 1))
                    P[a][s][s_next] += probs[a][i]

        return P

    @staticmethod
    def generate_R(demand: np.float64, n_states: np.int32, n_actions: np.int32) -> np.ndarray:
        """ Function that generates the rewards (costs) matrix.

        Devuelve una matriz C de dimensiones (n_actions x n_states x n_states),
        donde C[a, s, s'] = |demand - s'/100|. Si la acción aleja activamente
        al sistema de la demanda (increase llegando a un estado por encima de
        la demanda, o decrease llegando a uno por debajo) la distancia se
        multiplica por 2 para penalizar.
        """
        R = np.zeros((n_actions, n_states, n_states), dtype=np.float64)

        for a in range(n_actions):
            for s in range(n_states):
                for s_next in range(n_states):
                    nivel = s_next / 100.0
                    distancia = abs(demand - nivel)

                    # Penalizar x2 si la acción aleja del objetivo
                    if a == 2 and nivel > demand:    # increase alejándose por arriba
                        R[a][s][s_next] = 2.0 * distancia
                    elif a == 0 and nivel < demand:  # decrease alejándose por abajo
                        R[a][s][s_next] = 2.0 * distancia
                    else:
                        R[a][s][s_next] = distancia

        return R

    @staticmethod
    def control_iteration(demand: np.float64,
                          current_state: np.int32,
                          P: np.ndarray,
                          n_states: np.int32,
                          n_actions: np.int32,
                          gamma: np.float64) -> np.int32:
        """ Function that computes one control-iteration.

        Construye la matriz de costes para la demanda actual y resuelve el MDP
        mediante Iteración de Valores. Devuelve la acción óptima para el
        estado actual.
        """
        # Generar la matriz de costes para la demanda actual
        R = ControlModule.generate_R(demand, n_states, n_actions)

        # Crear y resolver el MDP con Value Iteration.
        # mdptoolbox MAXIMIZA recompensas, por lo que pasamos -R para que
        # equivalga a MINIMIZAR los costes (distancias a la demanda).
        mdp = mdptoolbox.mdp.ValueIteration(P, -R, gamma)
        mdp.run()

        # Devolver la acción óptima para el estado actual
        return mdp.policy[current_state]

    @staticmethod
    def control_loop(demand: np.ndarray,
                     probs: np.ndarray,
                     n_states: np.int32,
                     n_actions: np.int32,
                     gamma: np.float64) -> np.ndarray:
        """ Function that computes all the required iterations (control-loop) to satisfy the power demand.

        Para cada punto de la curva de demanda se resuelve un MDP, se obtiene
        la acción óptima y se aplica una transición estocástica al sistema de
        acuerdo con las probabilidades del reactor. La salida es la serie
        temporal de respuesta del reactor (potencia normalizada en [0, 1]).
        """
        # P es invariante durante todo el bucle (sólo dependen de las
        # probabilidades del reactor), así que se calcula una única vez.
        P = ControlModule.generate_P(probs, n_states, n_actions)

        # Inicializar el estado más cercano al primer punto de demanda
        current_state = int(np.clip(demand[0] * n_states, 0, n_states - 1))

        # Serie temporal de respuesta del reactor
        response = np.zeros(len(demand), dtype=np.float64)

        for t in range(len(demand)):
            # Obtener la acción óptima para este instante
            action = ControlModule.control_iteration(demand[t], current_state,
                                                     P, n_states, n_actions, gamma)

            # Simular la transición estocástica según las probabilidades del reactor
            delta = np.random.choice(ACTION_DELTAS[action], p=probs[action])

            # Actualizar estado con clip para no salirse de los bordes
            current_state = int(np.clip(current_state + delta, 0, n_states - 1))

            # Guardar la respuesta normalizada entre 0 y 1
            response[t] = current_state / n_states

        return response
