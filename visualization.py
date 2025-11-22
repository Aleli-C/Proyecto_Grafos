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
        line=dict(color="rgba(200,200,200,0.3)", width=1.5),
        hoverinfo="skip",
        showlegend=False,
        name="Grafo completo",
    )


def _build_nodes_trace(cities: List[dict], xs: List[float], ys: List[float], 
                       highlight_first: bool = False) -> go.Scatter:
    """
    Construye el trace de nodos con etiquetas de ciudades.

    Args:
        cities: Lista de diccionarios con la información de las ciudades.
        xs: Lista de coordenadas X de los nodos.
        ys: Lista de coordenadas Y de los nodos.
        highlight_first: Si True, resalta la primera ciudad (Denver) en color diferente.

    Returns:
        Trace Scatter con los nodos y sus etiquetas.
    """
    labels = [f"{i}: {c['name']}" for i, c in enumerate(cities)]
    
    if highlight_first:
        # Colores: primer nodo en verde brillante, resto en gris
        colors = ['rgb(46, 204, 113)'] + ['rgb(140,140,140)'] * (len(cities) - 1)
        sizes = [16] + [11] * (len(cities) - 1)
    else:
        colors = 'rgb(140,140,140)'
        sizes = 11
    
    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(color="rgb(255,255,255)", width=2),
        ),
        textfont=dict(color="rgb(40,40,40)", size=12, family="Arial Black"),
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

def animate_bruteforce_plotly(cities: List[dict], steps: list, units: str = "km"):
    """
    Genera una animación Plotly para la búsqueda exhaustiva del TSP.

    La animación muestra:
      - El grafo completo como fondo.
      - El ciclo actual en cada iteración (azul).
      - El mejor ciclo encontrado hasta ese momento (rojo).

    Args:
        cities: Lista de ciudades con coordenadas y nombre.
        steps: Lista de estados de la búsqueda. Cada elemento es:
               (path, length, best_path, best_length).
        units: Unidad de distancia para mostrar en títulos.
    """
    if not steps:
        return

    xs, ys = _compute_xy(cities)
    bg_trace = _build_background_trace(xs, ys)
    nodes_trace = _build_nodes_trace(cities, xs, ys, highlight_first=True)

    # Traces que se actualizan en cada frame
    current_trace = go.Scatter(
        x=[],
        y=[],
        mode="lines+markers",
        line=dict(color="rgb(52, 152, 219)", width=3),
        marker=dict(size=8, color="rgb(52, 152, 219)"),
        name="Ciclo actual",
    )
    best_trace = go.Scatter(
        x=[],
        y=[],
        mode="lines+markers",
        line=dict(color="rgb(231, 76, 60)", width=4),
        marker=dict(size=9, color="rgb(231, 76, 60)"),
        name="Mejor ciclo",
    )

    frames = []
    total = len(steps)

    for k, (path, length, best_path, best_length) in enumerate(steps):
        x_curr, y_curr = _path_to_xy(path, xs, ys)
        x_best, y_best = _path_to_xy(best_path, xs, ys)

        frame = go.Frame(
            data=[
                dict(type="scatter", x=x_curr, y=y_curr),
                dict(type="scatter", x=x_best, y=y_best),
            ],
            traces=[2, 3],
            name=str(k),
            layout=go.Layout(
                title=(
                    f"<b style='font-size:22px'>Búsqueda Exhaustiva (TSP)</b><br>"
                    f"<span style='font-size:18px'>Iteración {k+1}/{total} | "
                    f"L actual = {length:.4f} {units} · L* = {best_length:.4f} {units}</span>"
                )
            ),
        )
        frames.append(frame)

    fig = go.Figure(
        data=[bg_trace, nodes_trace, current_trace, best_trace],
        layout=go.Layout(
            title=f"<b style='font-size:22px'>Búsqueda Exhaustiva (TSP)</b>",
            width=1300,
            height=850,
            margin=dict(t=100, r=250, b=100, l=80),
            xaxis=dict(
                title="<b style='font-size:18px'>Longitud (escala relativa)</b>",
                showgrid=True,
                gridcolor="rgb(230,230,230)",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linewidth=2,
                linecolor="rgb(30,30,30)",
                mirror=True,
            ),
            yaxis=dict(
                title="<b style='font-size:18px'>Latitud (escala relativa)</b>",
                scaleanchor="x",
                scaleratio=1,
                showgrid=True,
                gridcolor="rgb(230,230,230)",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linewidth=2,
                linecolor="rgb(30,30,30)",
                mirror=True,
            ),
            plot_bgcolor="rgb(250,250,250)",
            paper_bgcolor="white",
            font=dict(color="rgb(30,30,30)", size=15, family="Arial"),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=True,
                    x=1.02,
                    y=0.5,
                    xanchor="left",
                    yanchor="middle",
                    direction="down",
                    pad=dict(r=10, t=10, b=10, l=10),
                    bgcolor="rgb(240,240,240)",
                    bordercolor="rgb(30,30,30)",
                    borderwidth=2,
                    buttons=[
                        dict(
                            label="<b>▶ PLAY</b>",
                            method="animate",
                            args=[None, {"frame": {"duration": 70, "redraw": True},
                                         "fromcurrent": True}],
                        ),
                        dict(
                            label="<b>⏸ PAUSE</b>",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate"}],
                        ),
                    ],
                )
            ],
            legend=dict(
                x=1.02,
                y=0.98,
                xanchor="left",
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="rgb(30,30,30)",
                borderwidth=2,
                font=dict(size=14),
            ),
        ),
        frames=frames,
    )

    # Mostrar en navegador para evitar dependencia nbformat
    fig.show(renderer="browser")


# ==============================
# Animación: Vecino Más Cercano
# ==============================

def animate_nearest_neighbor_plotly(cities: List[dict], steps: list, units: str = "km"):
    """
    Genera una animación Plotly para la heurística de Vecino Más Cercano.

    La animación muestra:
      - El grafo completo como fondo.
      - La ruta parcial construida hasta el momento (naranja).
      - La arista actual que se añade en cada paso (verde, grosor según distancia).

    Args:
        cities: Lista de ciudades con coordenadas y nombre.
        steps: Lista de estados de la heurística. Cada elemento es:
               (path, length, current, next_city, dist).
        units: Unidad de distancia para mostrar en títulos.
    """
    if not steps:
        return

    xs, ys = _compute_xy(cities)
    bg_trace = _build_background_trace(xs, ys)
    nodes_trace = _build_nodes_trace(cities, xs, ys, highlight_first=True)

    # Ruta parcial (naranja) y arista actual (verde)
    path_trace = go.Scatter(
        x=[],
        y=[],
        mode="lines+markers",
        line=dict(color="rgb(230, 126, 34)", width=3),
        marker=dict(size=8, color="rgb(230, 126, 34)"),
        name="Ruta parcial",
    )
    edge_trace = go.Scatter(
        x=[],
        y=[],
        mode="lines",
        line=dict(color="rgb(39, 174, 96)", width=5),
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
        width = 4 + 8 * t

        frame = go.Frame(
            data=[
                dict(type="scatter", x=x_path, y=y_path),
                dict(type="scatter", x=xe, y=ye, line=dict(width=width)),
            ],
            traces=[2, 3],
            name=str(k),
            layout=go.Layout(
                title=(
                    f"<b style='font-size:22px'>Heurística Vecino Más Cercano (NN)</b><br>"
                    f"<span style='font-size:18px'>Paso {k+1}/{total} | "
                    f"{cities[current]['name']} → {cities[next_city]['name']} "
                    f"(dist = {dist:.4f} {units}) · L acum = {length:.4f} {units}</span>"
                )
            ),
        )
        frames.append(frame)

    fig = go.Figure(
        data=[bg_trace, nodes_trace, path_trace, edge_trace],
        layout=go.Layout(
            title=f"<b style='font-size:22px'>Heurística Vecino Más Cercano (NN)</b>",
            width=1300,
            height=850,
            margin=dict(t=100, r=250, b=100, l=80),
            xaxis=dict(
                title="<b style='font-size:18px'>Longitud (escala relativa)</b>",
                showgrid=True,
                gridcolor="rgb(230,230,230)",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linewidth=2,
                linecolor="rgb(30,30,30)",
                mirror=True,
            ),
            yaxis=dict(
                title="<b style='font-size:18px'>Latitud (escala relativa)</b>",
                scaleanchor="x",
                scaleratio=1,
                showgrid=True,
                gridcolor="rgb(230,230,230)",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linewidth=2,
                linecolor="rgb(30,30,30)",
                mirror=True,
            ),
            plot_bgcolor="rgb(250,250,250)",
            paper_bgcolor="white",
            font=dict(color="rgb(30,30,30)", size=15, family="Arial"),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=True,
                    x=1.02,
                    y=0.5,
                    xanchor="left",
                    yanchor="middle",
                    direction="down",
                    pad=dict(r=10, t=10, b=10, l=10),
                    bgcolor="rgb(240,240,240)",
                    bordercolor="rgb(30,30,30)",
                    borderwidth=2,
                    buttons=[
                        dict(
                            label="<b>▶ PLAY</b>",
                            method="animate",
                            args=[None, {"frame": {"duration": 600, "redraw": True},
                                         "fromcurrent": True}],
                        ),
                        dict(
                            label="<b>⏸ PAUSE</b>",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate"}],
                        ),
                    ],
                )
            ],
            legend=dict(
                x=1.02,
                y=0.98,
                xanchor="left",
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="rgb(30,30,30)",
                borderwidth=2,
                font=dict(size=14),
            ),
        ),
        frames=frames,
    )

    # Mostrar en navegador para evitar dependencia nbformat
    fig.show(renderer="browser")


# ==============================
# Mapa estático 
# ==============================

def plot_static_map_plotly(
    cities: List[dict],
    opt_path: List[int],
    opt_length: float,
    nn_path: List[int],
    nn_length: float,
    units: str = "km",
):
    """
    Genera un mapa estático comparando la ruta óptima y la heurística NN.

    El mapa se compone de dos subgráficos:
      - Subgráfico 1: ruta óptima (búsqueda exhaustiva, en rojo).
      - Subgráfico 2: ruta generada por la heurística Vecino Más Cercano (naranja).

    Args:
        cities: Lista de ciudades con coordenadas y nombre.
        opt_path: Recorrido óptimo del TSP (lista de índices de ciudades).
        opt_length: Longitud total del recorrido óptimo.
        nn_path: Recorrido generado por la heurística NN.
        nn_length: Longitud total del recorrido NN.
        units: Unidad de distancia para mostrar en títulos.
    """
    xs, ys = _compute_xy(cities)

    def path_to_xy(path: List[int]):
        return _path_to_xy(path, xs, ys)

    # Calcular el gap de optimalidad
    gap = ((nn_length - opt_length) / opt_length * 100.0) if opt_length > 0 else 0

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"<b style='font-size:20px'>Ruta Óptima (Exhaustiva)</b><br>"
            f"<span style='font-size:18px'>L* = {opt_length:.4f} {units}</span><br>",
    
            f"<b style='font-size:20px'>Heurística Vecino Más Cercano</b><br>"
            f"<span style='font-size:18px'>L_NN = {nn_length:.4f} {units} ",
        ),

        horizontal_spacing=0.10,
        vertical_spacing=0.22,
    )

    # Función auxiliar para crear nodos con Denver resaltado
    def add_nodes_to_subplot(row, col, path):
        # Nodos normales (sin Denver)
        xs_regular = [xs[i] for i in range(len(cities)) if i != 0]
        ys_regular = [ys[i] for i in range(len(cities)) if i != 0]
        labels_regular = [f"{i}: {cities[i]['name']}" for i in range(len(cities)) if i != 0]
        
        fig.add_trace(
            go.Scatter(
                x=xs_regular,
                y=ys_regular,
                mode="markers+text",
                marker=dict(size=12, color="rgb(120,120,120)", 
                           line=dict(color="rgb(255,255,255)", width=2)),
                text=labels_regular,
                textposition="top center",
                textfont=dict(color="rgb(40,40,40)", size=12, family="Arial"),
                showlegend=False,
                hoverinfo="text",
            ),
            row=row,
            col=col,
        )
        
        # Denver (ciudad 0) resaltado en verde
        fig.add_trace(
            go.Scatter(
                x=[xs[0]],
                y=[ys[0]],
                mode="markers+text",
                marker=dict(size=18, color="rgb(46, 204, 113)", 
                           line=dict(color="rgb(255,255,255)", width=3),
                           symbol="star"),
                text=[f"0: {cities[0]['name']}<br>(INICIO)"],
                textposition="top center",
                textfont=dict(color="rgb(46, 204, 113)", size=13, family="Arial Black"),
                showlegend=False,
                hoverinfo="text",
            ),
            row=row,
            col=col,
        )

    # Ruta óptima (roja)
    x_opt, y_opt = path_to_xy(opt_path)
    fig.add_trace(
        go.Scatter(
            x=x_opt,
            y=y_opt,
            mode="lines+markers",
            line=dict(color="rgb(231, 76, 60)", width=4),
            marker=dict(size=10, color="rgb(231, 76, 60)", 
                       line=dict(color="rgb(255,255,255)", width=2)),
            name="Ruta óptima",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    add_nodes_to_subplot(1, 1, opt_path)

    # Ruta NN (naranja)
    x_nn, y_nn = path_to_xy(nn_path)
    fig.add_trace(
        go.Scatter(
            x=x_nn,
            y=y_nn,
            mode="lines+markers",
            line=dict(color="rgb(230, 126, 34)", width=4),
            marker=dict(size=10, color="rgb(230, 126, 34)",
                       line=dict(color="rgb(255,255,255)", width=2)),
            name="Ruta NN",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    add_nodes_to_subplot(1, 2, nn_path)

    # Ajuste de ejes con marco negro
    for col in [1, 2]:
        fig.update_xaxes(
            title_text="<b>Longitud (escala relativa)</b>",
            row=1,
            col=col,
            showgrid=True,
            gridcolor="rgb(230,230,230)",
            gridwidth=1,
            showline=True,
            linewidth=3,
            linecolor="rgb(30,30,30)",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Latitud (escala relativa)</b>",
            row=1,
            col=col,
            scaleanchor=f"x{col}",
            scaleratio=1,
            showgrid=True,
            gridcolor="rgb(230,230,230)",
            gridwidth=1,
            showline=True,
            linewidth=3,
            linecolor="rgb(30,30,30)",
            mirror=True,
        )

    fig.update_layout(
        title=(
            f"<b style='font-size:24px'>Problema del Viajante (TSP) - Comparación de Métodos</b><br>"
            f"<span style='font-size:18px'>Grafo Euclidiano con {len(cities)} ciudades</span>"
        ),
        showlegend=False,
        plot_bgcolor="rgb(250,250,250)",
        paper_bgcolor="white",
        width=1500,
        height=750,
        margin=dict(t=170, r=80, b=80, l=80),
        font=dict(color="rgb(30,30,30)", size=15, family="Arial"),
        title_y=0.95,
    )
    # Separar visualmente los títulos de los subplots del gráfico
    for ann in fig['layout']['annotations']:
        ann['yshift'] = 20

    # Mostrar en navegador para evitar dependencia nbformat
    fig.show(renderer="browser")