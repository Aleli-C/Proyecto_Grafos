# visualization.py
from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==============================
# Utilidades internas
# ==============================

def _compute_xy(cities: List[dict]):
    """
    Calcula coordenadas reescaladas a partir de latitud/longitud.

    Centra los puntos en torno al promedio y aplica una escala lineal
    para que el grafo quede más grande y distribuido, respetando las
    proporciones euclidianas.

    Args:
        cities: Lista de diccionarios con claves "lat" y "lon".

    Returns:
        Tuple (xs, ys) con las coordenadas reescaladas.
    """
    xs_raw = [c["lon"] for c in cities]
    ys_raw = [c["lat"] for c in cities]

    n = len(cities)
    mean_x = sum(xs_raw) / n
    mean_y = sum(ys_raw) / n

    # Factor de escala para separar más los nodos en la figura
    scale = 5.0

    xs = [(x - mean_x) * scale for x in xs_raw]
    ys = [(y - mean_y) * scale for y in ys_raw]
    return xs, ys


def _build_background_trace(xs: List[float], ys: List[float]) -> go.Scatter:
    """
    Construye el trace de fondo con todas las aristas del grafo completo.

    Cada arista se dibuja como una línea gris clara, separando los
    segmentos con None para que Plotly los interprete como trazos
    independientes.

    Args:
        xs: Lista de coordenadas X de los nodos.
        ys: Lista de coordenadas Y de los nodos.

    Returns:
        Trace Scatter con todas las aristas del grafo completo.
    """
    x_bg = []
    y_bg = []
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            x_bg.extend([xs[i], xs[j], None])
            y_bg.extend([ys[i], ys[j], None])

    return go.Scatter(
        x=x_bg,
        y=y_bg,
        mode="lines",
        line=dict(color="rgba(160,160,160,0.5)", width=1.8),  # más gruesas
        hoverinfo="skip",
        showlegend=False,
        name="Grafo completo",
    )


def _build_nodes_trace(cities: List[dict], xs: List[float], ys: List[float]) -> go.Scatter:
    """
    Construye el trace de nodos con etiquetas de ciudades.

    Args:
        cities: Lista de diccionarios con la información de las ciudades.
        xs: Lista de coordenadas X de los nodos.
        ys: Lista de coordenadas Y de los nodos.

    Returns:
        Trace Scatter con los nodos y sus etiquetas.
    """
    labels = [f"{i}: {c['name']}" for i, c in enumerate(cities)]
    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(
            size=11,
            color="rgb(140,140,140)",
            line=dict(color="rgb(80,80,80)", width=1.2),
        ),
        textfont=dict(color="rgb(60,60,60)", size=11),
        name="Ciudades",
        hoverinfo="text",
    )


def _path_to_xy(path: List[int], xs: List[float], ys: List[float]):
    """
    Convierte un recorrido de índices de nodos a coordenadas X, Y.

    Args:
        path: Lista de índices de nodos.
        xs: Lista de coordenadas X de todos los nodos.
        ys: Lista de coordenadas Y de todos los nodos.

    Returns:
        Tuple (x, y) con las coordenadas del recorrido.
    """
    x = [xs[i] for i in path]
    y = [ys[i] for i in path]
    return x, y


# ==============================
# Animación: búsqueda exhaustiva
# ==============================

def animate_bruteforce_plotly(cities: List[dict], steps: list):
    """
    Genera una animación Plotly para la búsqueda exhaustiva del TSP.

    La animación muestra:
      - El grafo completo como fondo.
      - El ciclo actual en cada iteración (naranjo).
      - El mejor ciclo encontrado hasta ese momento (rojo).

    Args:
        cities: Lista de ciudades con coordenadas y nombre.
        steps: Lista de estados de la búsqueda. Cada elemento es:
               (path, length, best_path, best_length).
    """
    if not steps:
        return

    xs, ys = _compute_xy(cities)
    bg_trace = _build_background_trace(xs, ys)
    nodes_trace = _build_nodes_trace(cities, xs, ys)

    # Traces que se actualizan en cada frame
    current_trace = go.Scatter(
        x=[],
        y=[],
        mode="lines+markers",
        line=dict(color="orange", width=2.6),  # un poco más gruesa
        marker=dict(size=7),
        name="Ciclo actual",
    )
    best_trace = go.Scatter(
        x=[],
        y=[],
        mode="lines+markers",
        line=dict(color="red", width=3.4),     # un poco más gruesa
        marker=dict(size=7),
        name="Mejor ciclo",
    )

    frames = []
    total = len(steps)

    for k, (path, length, best_path, best_length) in enumerate(steps):
        x_curr, y_curr = _path_to_xy(path, xs, ys)
        x_best, y_best = _path_to_xy(best_path, xs, ys)

        frame = go.Frame(
            data=[
                # Actualiza los traces en posiciones 2 (ciclo actual) y 3 (mejor ciclo)
                dict(type="scatter", x=x_curr, y=y_curr),
                dict(type="scatter", x=x_best, y=y_best),
            ],
            traces=[2, 3],
            name=str(k),
            layout=go.Layout(
                title=(
                    f"Búsqueda exhaustiva (TSP) | "
                    f"Iteración {k+1}/{total} | "
                    f"L actual = {length:.4f} · L* = {best_length:.4f}"
                )
            ),
        )
        frames.append(frame)

    fig = go.Figure(
        data=[bg_trace, nodes_trace, current_trace, best_trace],
        layout=go.Layout(
            title="Búsqueda exhaustiva (TSP)",
            width=1200,
            height=900,
            margin=dict(t=90, r=60, b=80, l=70),
            xaxis=dict(
                title="Longitud (escala relativa)",
                showgrid=True,
                gridcolor="rgb(220,220,220)",
                zeroline=False,
            ),
            yaxis=dict(
                title="Latitud (escala relativa)",
                scaleanchor="x",   # Mantiene proporción 1:1
                scaleratio=1,
                showgrid=True,
                gridcolor="rgb(220,220,220)",
                zeroline=False,
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="black", size=13),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.5,
                    y=0.02,               # abajo
                    xanchor="center",
                    yanchor="bottom",
                    direction="right",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 70, "redraw": True},
                                         "fromcurrent": True}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate"}],
                        ),
                    ],
                )
            ],
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
        ),
        frames=frames,
    )

    fig.show()


# ==============================
# Animación: Vecino Más Cercano
# ==============================

def animate_nearest_neighbor_plotly(cities: List[dict], steps: list):
    """
    Genera una animación Plotly para la heurística de Vecino Más Cercano.

    La animación muestra:
      - El grafo completo como fondo.
      - La ruta parcial construida hasta el momento (naranjo).
      - La arista actual que se añade en cada paso (rojo, grosor según distancia).

    Args:
        cities: Lista de ciudades con coordenadas y nombre.
        steps: Lista de estados de la heurística. Cada elemento es:
               (path, length, current, next_city, dist).
    """
    if not steps:
        return

    xs, ys = _compute_xy(cities)
    bg_trace = _build_background_trace(xs, ys)
    nodes_trace = _build_nodes_trace(cities, xs, ys)

    # Ruta parcial (naranjo) y arista actual (rojo)
    path_trace = go.Scatter(
        x=[],
        y=[],
        mode="lines+markers",
        line=dict(color="orange", width=2.6),
        marker=dict(size=7),
        name="Ruta parcial",
    )
    edge_trace = go.Scatter(
        x=[],
        y=[],
        mode="lines",
        line=dict(color="red", width=4.5),
        name="Arista actual",
    )

    # Se usa la distancia de la arista actual para ajustar el grosor
    dists = [step[4] for step in steps]
    min_d = min(dists)
    max_d = max(dists) if max(dists) > min_d else min_d + 1e-9

    frames = []
    total = len(steps)

    for k, (path, length, current, next_city, dist) in enumerate(steps):
        x_path, y_path = _path_to_xy(path, xs, ys)
        xe = [xs[current], xs[next_city]]
        ye = [ys[current], ys[next_city]]

        # Normaliza el grosor de la arista actual según la distancia
        t = (dist - min_d) / (max_d - min_d)
        width = 3 + 7 * t  # un poco más base y más rango

        frame = go.Frame(
            data=[
                dict(type="scatter", x=x_path, y=y_path),
                dict(type="scatter", x=xe, y=ye, line=dict(width=width)),
            ],
            traces=[2, 3],
            name=str(k),
            layout=go.Layout(
                title=(
                    "Heurística Vecino Más Cercano (NN) | "
                    f"Paso {k+1}/{total} | "
                    f"{cities[current]['name']} → {cities[next_city]['name']} "
                    f"(dist = {dist:.4f}) · L acum = {length:.4f}"
                )
            ),
        )
        frames.append(frame)

    fig = go.Figure(
        data=[bg_trace, nodes_trace, path_trace, edge_trace],
        layout=go.Layout(
            title="Heurística Vecino Más Cercano (NN)",
            width=1200,
            height=900,
            margin=dict(t=90, r=60, b=80, l=70),
            xaxis=dict(
                title="Longitud (escala relativa)",
                showgrid=True,
                gridcolor="rgb(220,220,220)",
                zeroline=False,
            ),
            yaxis=dict(
                title="Latitud (escala relativa)",
                scaleanchor="x",
                scaleratio=1,
                showgrid=True,
                gridcolor="rgb(220,220,220)",
                zeroline=False,
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="black", size=13),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.5,
                    y=0.02,              # abajo
                    xanchor="center",
                    yanchor="bottom",
                    direction="right",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 600, "redraw": True},
                                         "fromcurrent": True}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate"}],
                        ),
                    ],
                )
            ],
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
        ),
        frames=frames,
    )

    fig.show()


# ==============================
# Mapa estático 
# ==============================

def plot_static_map_plotly(
    cities: List[dict],
    opt_path: List[int],
    opt_length: float,
    nn_path: List[int],
    nn_length: float,
):
    """
    Genera un mapa estático comparando la ruta óptima y la heurística NN.

    El mapa se compone de dos subgráficos:
      - Subgráfico 1: ruta óptima (búsqueda exhaustiva, en rojo).
      - Subgráfico 2: ruta generada por la heurística Vecino Más Cercano (naranjo).

    Args:
        cities: Lista de ciudades con coordenadas y nombre.
        opt_path: Recorrido óptimo del TSP (lista de índices de ciudades).
        opt_length: Longitud total del recorrido óptimo.
        nn_path: Recorrido generado por la heurística NN.
        nn_length: Longitud total del recorrido NN.
    """
    xs, ys = _compute_xy(cities)

    def path_to_xy(path: List[int]):
        return _path_to_xy(path, xs, ys)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Ruta óptima (exhaustiva) - L* = {opt_length:.4f}",
            f"Heurística Vecino Más Cercano - L_NN = {nn_length:.4f}",
        ),
    )

    # Ruta óptima (roja)
    x_opt, y_opt = path_to_xy(opt_path)
    fig.add_trace(
        go.Scatter(
            x=x_opt,
            y=y_opt,
            mode="lines+markers+text",
            line=dict(color="red", width=3.5),
            marker=dict(size=9, color="rgb(120,120,120)"),
            text=[cities[i]["name"] for i in opt_path],
            textposition="top center",
            name="Ruta óptima",
        ),
        row=1,
        col=1,
    )

    # Ruta NN (naranjo)
    x_nn, y_nn = path_to_xy(nn_path)
    fig.add_trace(
        go.Scatter(
            x=x_nn,
            y=y_nn,
            mode="lines+markers+text",
            line=dict(color="orange", width=3.5),
            marker=dict(size=9, color="rgb(120,120,120)"),
            text=[cities[i]["name"] for i in nn_path],
            textposition="top center",
            name="Ruta NN",
        ),
        row=1,
        col=2,
    )

    # Ajuste de ejes (mantiene proporción entre ejes X e Y)
    fig.update_xaxes(title_text="Longitud (escala relativa)", row=1, col=1)
    fig.update_yaxes(
        title_text="Latitud (escala relativa)",
        row=1,
        col=1,
        scaleanchor="x1",
        scaleratio=1,
    )
    fig.update_xaxes(title_text="Longitud (escala relativa)", row=1, col=2)
    fig.update_yaxes(
        title_text="Latitud (escala relativa)",
        row=1,
        col=2,
        scaleanchor="x2",
        scaleratio=1,
    )

    fig.update_layout(
        title="TSP sobre grafo euclidiano (posiciones reescaladas)",
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1200,
        height=600,
        margin=dict(t=90, r=60, b=60, l=70),
        font=dict(color="black", size=13),
    )

    fig.show()
