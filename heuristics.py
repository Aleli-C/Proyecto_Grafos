# heuristics.py
import itertools
from typing import List, Tuple


def brute_force_tsp(D: List[List[float]], start: int = 0) -> Tuple[List[int], float, List[tuple]]:
    """
    Fuerza bruta para el problema del viajante (TSP).

    Recorre todas las permutaciones posibles fijando 'start' como origen
    y calcula el ciclo de menor longitud.

    Args:
        D: Matriz de distancias D[i][j] (coste de ir de i a j).
        start: Índice de la ciudad origen.

    Returns:
        best_path: Lista de índices de ciudades que forman el ciclo óptimo,
                   incluyendo el regreso al origen.
        best_length: Longitud total del ciclo óptimo.
        steps: Lista de estados para animación. Cada elemento es:
               (path_actual, longitud_actual, mejor_path_hasta_ahora, mejor_longitud_hasta_ahora).
    """
    n = len(D)
    other_nodes = [i for i in range(n) if i != start]

    # Caso trivial: solo existe la ciudad 'start'
    if not other_nodes:
        best_path = (start, start)
        best_length = 0.0
        steps = [(best_path, best_length, best_path, best_length)]
        return list(best_path), best_length, steps

    # Inicializamos con la primera permutación para evitar best_path = None
    perms_iter = itertools.permutations(other_nodes)
    first_perm = next(perms_iter)

    best_path = (start,) + first_perm + (start,)
    best_length = 0.0
    for i in range(len(best_path) - 1):
        best_length += D[best_path[i]][best_path[i + 1]]

    steps: List[tuple] = [(best_path, best_length, best_path, best_length)]

    # Recorremos el resto de permutaciones
    for perm in perms_iter:
        path = (start,) + perm + (start,)
        length = 0.0
        for i in range(len(path) - 1):
            length += D[path[i]][path[i + 1]]

        if length < best_length:
            best_length = length
            best_path = path

        steps.append((path, length, best_path, best_length))

    return list(best_path), best_length, steps


def nearest_neighbor_tsp(D: List[List[float]], start: int = 0) -> Tuple[List[int], float, List[tuple]]:
    """
    Heurística de Vecino Más Cercano para TSP.

    Construye una ruta empezando en 'start' y, en cada paso, elige
    la ciudad no visitada más cercana.

    Args:
        D: Matriz de distancias D[i][j] (coste de ir de i a j).
        start: Índice de la ciudad origen.

    Returns:
        path: Lista de índices que representan el ciclo generado
              (incluye el regreso a 'start').
        length: Longitud total del ciclo construido por la heurística.
        steps: Lista de estados para animación. Cada elemento es:
               (ruta_parcial, longitud_acumulada, ciudad_actual, siguiente_ciudad, distancia_entre_ellas).
    """
    n = len(D)
    unvisited = set(range(n))
    unvisited.remove(start)

    path: List[int] = [start]
    length = 0.0
    steps: List[tuple] = []
    current = start

    while unvisited:
        next_city = min(unvisited, key=lambda j: D[current][j])
        dist = D[current][next_city]

        length += dist
        path.append(next_city)
        steps.append((path.copy(), length, current, next_city, dist))

        unvisited.remove(next_city)
        current = next_city

    # Regreso al origen
    length += D[current][start]
    path.append(start)
    steps.append((path.copy(), length, current, start, D[current][start]))

    return path, length, steps
