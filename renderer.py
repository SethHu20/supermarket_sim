from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

import numpy as np
import matplotlib.pyplot as plt

from simulator import AisleModel, WallAgent, CustomerAgent

MODEL_HEIGHT = 4

MODEL_WIDTH = 11


def plot_examples(colormaps):
    """
    Helper function to plot data with associated colormap.
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()


def agent_portrayal(agent):
    portrayal = {
        "Filled": "true",
        "Layer": 0
    }
    if isinstance(agent, WallAgent):
        portrayal["Shape"] = "rect"
        portrayal["w"] = "1"
        portrayal["h"] = "1"
        agent: WallAgent
        if agent.type == 0:
            portrayal["Color"] = "black"
        if agent.type == 1:
            portrayal["Color"] = "red"
        if agent.type == 2:
            portrayal["Color"] = "green"
        if agent.type == 3:
            portrayal["Color"] = "blue"
    elif isinstance(agent, CustomerAgent):
        portrayal["Shape"] = "circle"
        portrayal["r"] = "1"
        portrayal["text"] = agent.unique_id
        portrayal["text_color"] = "white"
        agent: CustomerAgent
        if agent.demand == 0:
            portrayal["Color"] = "gray"
        if agent.demand == 1:
            portrayal["Color"] = "red"
        if agent.demand == 2:
            portrayal["Color"] = "green"
        if agent.demand == 3:
            portrayal["Color"] = "blue"
    # portrayal = {"Shape": "circle",
    #              "Filled": "true",
    #              "r": 0.5}
    return portrayal


grid = CanvasGrid(agent_portrayal, MODEL_WIDTH, MODEL_HEIGHT, MODEL_WIDTH * 50, MODEL_HEIGHT * 50)

# chart = ChartModule([{"Label": "Gini",
#                       "Color": "Black"}],
#                     data_collector_name='datacollector')

UserSettableParameter('slider', 'Red Chance', value=0.3, min_value=0, max_value=1, step=0.01)
UserSettableParameter('slider', 'Blue Chance', value=0.3, min_value=0, max_value=1, step=0.01)

chart1 = ChartModule([{"Label": "total_population",
                       "Color": "Black"},
                      {"Label": "G1_population",
                       "Color": "Red"},
                      {"Label": "G2_population",
                       "Color": "Green"},
                      {"Label": "G3_population",
                       "Color": "Blue"}
                      ],
                     data_collector_name='datacollector')

chart2 = ChartModule([{"Label": "current_population",
                       "Color": "gray"}],
                     data_collector_name='datacollector')

server = ModularServer(AisleModel,
                       [grid, chart1, chart2],
                       "Supermarket model",
                       {
                           "n": UserSettableParameter('slider', 'Customer n', value=100, min_value=1, max_value=200,
                                                      step=1),
                           "width": MODEL_WIDTH,
                           "height": MODEL_HEIGHT,
                           "spawn_chance": UserSettableParameter('slider', 'Spawn Chance', value=0.3, min_value=0,
                                                                 max_value=1,
                                                                 step=0.01),
                           "shelf_config": [0, 3, 3, 3, 1, 1, 1, 2, 2, 2, 0],
                           "probability_table": [0.6, 0.2, 0.2],
                           "seed": 0,
                       })

server.port = 8521  # The default
server.launch()
