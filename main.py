# main.py
import math
import time
import random

from dijkstra_alg import dijkstra, reconstruct_path
from heuristics import brute_force_tsp, nearest_neighbor_tsp
from visualization import (
    animate_bruteforce_plotly,
    animate_nearest_neighbor_plotly,
    plot_static_map_plotly,
)


# Coordenadas en formato decimal (lat, lon).
ALL_CITIES = [
    {"name": "Los Angeles",   "lat": 34.052235,  "lon": -118.243683},
    {"name": "San Francisco", "lat": 37.774929,  "lon": -122.419416},
    {"name": "Seattle",       "lat": 47.606209,  "lon": -122.332069},
    {"name": "Denver",        "lat": 39.739236,  "lon": -104.990251},
    {"name": "Chicago",       "lat": 41.878113,  "lon": -87.629799},
    {"name": "Houston",       "lat": 29.760427,  "lon": -95.369804},
    {"name": "Miami",         "lat": 25.761681,  "lon": -80.191788},
    {"name": "New York",      "lat": 40.712776,  "lon": -74.005974},
    {"name": "Atlanta",       "lat": 33.748995,  "lon": -84.387982},
]

# ==============================
# Utilidades de distancias
# ==============================

def euclidean_distance(p1: dict, p2: dict) -> float:
    """Distancia euclidiana simple entre dos puntos (lat, lon)."""
    return math.sqrt((p1["lat"] - p2["lat"]) ** 2 + (p1["lon"] - p2["lon"]) ** 2)


def build_distance_matrix(cities: list) -> list:
    """Construye la matriz D[i][j] de distancias euclidianas sobre la lista dada de ciudades."""
    n = len(cities)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i][j] = euclidean_distance(cities[i], cities[j])
            else:
                D[i][j] = 0.0
    return D


def print_distance_matrix(D: list, cities: list) -> None:
    """Imprime la matriz de distancias con nombres de ciudades."""
    n = len(D)
    print("Matriz de distancias (euclidiana sobre lat/long):")
    header = ["{:>12}".format(c["name"][:10]) for c in cities]
    print("{:>12}".format("") + "".join(header))
    for i in range(n):
        row = ["{:>12.4f}".format(D[i][j]) for j in range(n)]
        print("{:>12}".format(cities[i]["name"][:10]) + "".join(row))
    print()


# ==============================
# Selección de subconjunto de ciudades
# ==============================

def select_cities(cities: list, n: int = 9, seed: int = 42) -> list:
    """
    Selecciona n ciudades de la lista completa asegurando que 'Denver'
    sea el primer elemento (ciudad de inicio).

    Se usa un seed fijo para que el subconjunto sea reproducible.
    """
    if n > len(cities):
        raise ValueError("n no puede ser mayor al número de ciudades disponibles")
    # Buscar la ciudad "Denver" (case-insensitive)
    start_city = "denver"
    start_idx = next((i for i, c in enumerate(cities) if c["name"].lower() == start_city), None)
    if start_idx is None:
        raise ValueError("La ciudad 'Denver' no está en la lista de ciudades disponibles")

    # Si se pide sólo 1 ciudad, devolver solo Denver
    if n == 1:
        return [cities[start_idx]]

    # Seleccionar aleatoriamente el resto, sin incluir Denver, y mantener reproducibilidad
    remaining = [c for i, c in enumerate(cities) if i != start_idx]
    random.seed(seed)
    others = random.sample(remaining, n - 1)

    # Devolver Denver como primera ciudad (inicio) seguida por el resto seleccionadas
    return [cities[start_idx]] + others


# ==============================
# Ejecución principal
# ==============================

def main():
    # Seleccionar 9 ciudades de la base completa (15)
    cities = select_cities(ALL_CITIES, n=9)

    total = len(ALL_CITIES)
    n = len(cities)
    print(f"Número de ciudades seleccionadas para el experimento: {n}\n")

    print("Ciudades seleccionadas:")
    for i, c in enumerate(cities):
        print(
            f"  {i}: {c['name']} "
            f"(lat={c['lat']:.6f}, lon={c['lon']:.6f})"
        )
    print()

    D = build_distance_matrix(cities)
    print_distance_matrix(D, cities)

    # --- Dijkstra: ejemplo desde la ciudad índice 0 ---
    origen = 0
    dist, prev = dijkstra(D, origen)

    print("=== DIJKSTRA DESDE ORIGEN ===")
    for i, d_val in enumerate(dist):
        print(
            f"Distancia mínima {cities[origen]['name']} "
            f"-> {cities[i]['name']}: {d_val:.4f}"
        )
    print()

    # Camino mínimo desde origen hasta la última ciudad del subconjunto
    destino = n - 1
    path_dij = reconstruct_path(prev, origen, destino)
    if path_dij:
        names_path = " -> ".join(cities[i]["name"] for i in path_dij)
        print(
            f"Camino mínimo {cities[origen]['name']} "
            f"-> {cities[destino]['name']}:"
        )
        print("  Índices:", path_dij)
        print("  Nombres:", names_path)
        print(f"  Longitud total: {dist[destino]:.4f}\n")
    else:
        print(
            f"No hay camino desde {cities[origen]['name']} "
            f"a {cities[destino]['name']}\n"
        )

    # --- Búsqueda exhaustiva (óptimo TSP) ---
    t0 = time.perf_counter()
    opt_path, opt_length, steps_brute = brute_force_tsp(D, start=0)
    t1 = time.perf_counter()
    time_brute = t1 - t0

    print("=== RESULTADO BÚSQUEDA EXHAUSTIVA ===")
    print("Ruta óptima (índices):", opt_path)
    print("Ruta óptima (nombres):", " -> ".join(cities[i]["name"] for i in opt_path))
    print(f"Longitud óptima L* = {opt_length:.6f}")
    print(f"Tiempo de ejecución: {time_brute:.4f} segundos\n")

    # --- Heurística Vecino Más Cercano ---
    t0 = time.perf_counter()
    nn_path, nn_length, steps_nn = nearest_neighbor_tsp(D, start=0)
    t1 = time.perf_counter()
    time_nn = t1 - t0

    print("=== RESULTADO VECINO MÁS CERCANO ===")
    print("Ruta NN (índices):", nn_path)
    print("Ruta NN (nombres):", " -> ".join(cities[i]["name"] for i in nn_path))
    print(f"Longitud L_NN = {nn_length:.6f}")
    print(f"Tiempo de ejecución: {time_nn:.4f} segundos\n")

    # --- Gap de optimalidad ---
    gap = (nn_length - opt_length) / opt_length * 100.0
    print("=== COMPARACIÓN ===")
    print(f"Gap de optimalidad g = {gap:.2f} %")
    ratio = time_brute / time_nn if time_nn > 0 else float("inf")
    print(f"Relación de tiempos (exhaustivo / NN) = {ratio:.2f}x\n")

    # Unidad para las visualizaciones 
    units = " "

    # --- Mapa estático (Plotly) ---
    plot_static_map_plotly(cities, opt_path, opt_length, nn_path, nn_length, units=units)

    # --- Animaciones (Plotly) ---
    animate_bruteforce_plotly(cities, steps_brute, units=units)
    animate_nearest_neighbor_plotly(cities, steps_nn, units=units)


if __name__ == "__main__":
    main()