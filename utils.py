from tetris_main import Population
import plotly.graph_objects as go
import numpy as np
import plotly.express as px


def plot_grid(
    grid,
    pieces_coordinates: list,
    ind_fitness = None,
    save_html_name: str = None,
    save_png_name: str = None,
    marker_size: int = 50,
    width: int = 600,
    height: int = 600,
):
    """Generate interactive plot with pieces placed.

    Args:
        population (Population): population
        grid: Grid filled
        pieces_coordinates (list): coordinates of each piece in grid
        save_html_name (str, optional): If path given, saves to file (do not include file). Defaults to None.
        save_png_name (str, optional): If name given, saves to file with scale 2. Defaults to None.
        marker_size (int, optional): Increase or decrease marker size depending on grid size. Defaults to 50.
        width (int, optional): Width. Defaults to 600.
        height (int, optional): Height. Defaults to 600.
    """
    if ind_fitness != None:
        title = f"<b>Fitness:</b> {int(ind_fitness)}/{int(np.sum(100*np.ones_like(grid)))}<br><b>Pieces used:</b> {len(pieces_coordinates)}/{grid.size//4}"
    else:
        title = ""
    fig = go.Figure(
        data=[
            go.Scatter(
                x=pieces_coordinates[i][:, 1],
                y=pieces_coordinates[i][:, 0],
                marker_size=marker_size,
                marker_symbol="square",
                mode="markers",
            )
            for i in range(len(pieces_coordinates))
        ],
        layout=dict(
            margin=dict(l=00, r=00, t=80, b=0),
            width=width,
            height=height,
            title=title,
            xaxis=dict(
                range=[-0.5, grid.shape[1] - 0.5], fixedrange=True, showticklabels=False
            ),
            yaxis=dict(
                range=[grid.shape[0] - 0.5, -0.5],
                tickfont_size=12,
                showticklabels=False,
            ),
        ),
    )
    fig.update_layout(showlegend=False)
    if save_html_name != None:
        fig.write_html(f"{save_html_name}.html")
    if save_png_name != None:
        fig.write_image(f"{save_png_name}.png", scale=2)

    fig.show()


def plot_fitness(fitness_history, save_html_name=None, save_png_name=None):
    fig = px.line(
        fitness_history,
        title="",
        labels={"index": "Generation", "value": "Fitness"},
        width=600,
        height=300,
    )
    fig.update_layout(yaxis_tickformat="0", margin=dict(l=60, r=40, t=20, b=0))
    if save_html_name != None:
        fig.write_html(f"{save_html_name}.html")
    if save_png_name != None:
        fig.write_image(f"{save_png_name}.png", scale=2)
    fig.show()
