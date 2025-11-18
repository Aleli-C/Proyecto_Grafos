# dijkstra_alg.py
import math
from typing import List, Tuple, Optional


def dijkstra(D: List[List[float]], start: int) -> Tuple[List[float], List[Optional[int]]]:
    """
    Implementa el algoritmo de Dijkstra sobre una matriz de adyacencia.

    Args:
        D: Matriz de distancias, donde D[u][v] >= 0 representa el costo de la
           arista (u, v). Se puede usar math.inf para indicar ausencia de arista.
        start: Índice del nodo origen.

    Returns:
        dist: Lista con la distancia mínima desde 'start' a cada nodo.
        prev: Lista de predecesores para reconstruir los caminos mínimos.
    """
    n = len(D)
    dist = [math.inf] * n
    prev: List[Optional[int]] = [None] * n
    visited = [False] * n

    dist[start] = 0.0

    for _ in range(n):
        # Selecciona el nodo no visitado con menor distancia provisional
        u = None
        best = math.inf
        for i in range(n):
            if not visited[i] and dist[i] < best:
                best = dist[i]
                u = i

        # No quedan nodos alcanzables desde 'start'
        if u is None:
            break

        visited[u] = True

        # Relaja las aristas salientes de u
        for v in range(n):
            w = D[u][v]
            if w <= 0 or w == math.inf:
                continue
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u

    return dist, prev


def reconstruct_path(prev: List[Optional[int]], start: int, goal: int) -> list:
    """
    Reconstruye el camino mínimo desde 'start' hasta 'goal'
    usando la lista de predecesores generada por Dijkstra.

    Args:
        prev: Lista de predecesores devuelta por dijkstra.
        start: Índice del nodo origen.
        goal: Índice del nodo destino.

    Returns:
        Lista de índices de nodos que representan el camino mínimo
        desde 'start' hasta 'goal'. Si no existe camino, retorna [].
    """
    path = []
    current = goal

    while current is not None:
        path.append(current)
        if current == start:
            break
        current = prev[current]

    # No hay camino entre start y goal
    if not path or path[-1] != start:
        return []

    path.reverse()
    return path
