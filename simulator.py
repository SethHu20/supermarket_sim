from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler
from mesa.space import MultiGrid, SingleGrid
from mesa.datacollection import DataCollector
from matplotlib.colors import ListedColormap
from mesa import batchrunner
import matplotlib.pyplot as plt
import numpy as np
import seaborn


class WallAgent(Agent):

    def __init__(self, unique_id, model, product_type):
        super().__init__(unique_id, model)
        self.type = product_type


class CustomerAgent(Agent):

    def __init__(self, unique_id, model, demand_type):
        super().__init__(unique_id, model)
        self.demand = demand_type

    def step(self) -> None:
        self.model: AisleModel
        move_list = []
        if self.demand != 0:
            for neighbour in self.model.grid.get_neighbors(self.pos, False, ):
                if isinstance(neighbour, WallAgent):
                    if neighbour.type == self.demand:
                        self.demand = 0
            x, y = self.pos
            x: int
            y: int
            if self.model.grid.is_cell_empty((x, y-1)):
                self.model.grid.move_agent(self, (x, y-1))
            elif self.model.grid.is_cell_empty((x+1, y)):
                self.model.grid.move_agent(self, (x+1, y))
            else:
                self.model.grid.move_to_empty(self)
        elif self.demand == 0:
            if self.pos in self.model.exit:
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)
            else:
                x, y = self.pos
                x: int
                y: int
                if self.model.grid.is_cell_empty((x, y + 1)):
                    self.model.grid.move_agent(self, (x, y + 1))
                elif self.model.grid.is_cell_empty((x - 1, y)):
                    self.model.grid.move_agent(self, (x - 1, y))



class AisleModel(Model):

    def __init__(
            self,
            N,
            width=11,
            height=4,
            probability_table=None,
            spawn_chance=0.3,
            seed=0
    ):
        super().__init__()
        self.grid = SingleGrid(width, height, False)
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.running = True
        self.spawn_chance = spawn_chance

        if probability_table is None:
            probability_table = [0.333, 0.333, 0.334]
        elif sum(probability_table) != 1:
            return Exception("Probability Table not proper")
        self.probability_table = probability_table

        # wall creation
        wall_config = [0,1,1,1,2,2,2,3,3,3,0]

        for i in range(width):
            # wall = WallAgent(None, self, 0)
            # self.grid.position_agent(wall, i, 0)
            wall = WallAgent(None, self, 0)
            self.grid.position_agent(wall, i, height - 1)

        for i, target_type in enumerate(wall_config):
            wall = WallAgent(None, self, target_type)
            # self.grid.remove_agent(self.grid.get_cell_list_contents((i, 0)))
            self.grid.position_agent(wall, i, 0)

        for i in range(1, height-1):
            wall = WallAgent(None, self, 0)
            self.grid.position_agent(wall, width - 1, i)

        self.entrance = [(0, 2)]
        self.exit = [(0, 1)] + self.entrance

        self.datacollector = DataCollector(
            model_reporters={"population": current_population},
            tables={"Lifespan": ["unique_id", "age"]}
        )

    def rand_type(self) -> int:
        rng = self.random.random()
        for product_type, probability in enumerate(self.probability_table, 1):
            if rng <= probability:
                return product_type
            else:
                rng -= probability
        return len(self.probability_table) - 1

    def step(self) -> None:
        self.datacollector.collect(self)
        self.schedule.step()
        if self.random.random() < self.spawn_chance:
            for potential_entrance in self.exit:
                if self.grid.is_cell_empty(potential_entrance):
                    customer = CustomerAgent(self.next_id(), self, self.rand_type())
                    self.grid.position_agent(customer, *potential_entrance)
                    self.schedule.add(customer)
                    break


def current_population(model: AisleModel) -> int:
    pass


def main4():
    model = AisleModel(10)

    grid_point = np.zeros((model.grid.width, model.grid.height))

    for cell, x, y in model.grid.coord_iter():
        if isinstance(cell, WallAgent):
            cell: WallAgent
            grid_point[x][y] = cell.type
        else:
            grid_point[x][y] = -1

    grid_point = grid_point.transpose()

    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(model.grid.width), minor=False)
    ax.set_yticks(np.arange(model.grid.height), minor=False)

    cmap = ListedColormap(['white', 'black', 'red', 'green', 'blue'])

    ax.imshow(grid_point, interpolation="nearest", origin='upper', extent=(0, model.grid.width, 0, model.grid.height), cmap=cmap)
    # psm = ax.pcolormesh(grid_point, rasterized=True, vmin=-4, vmax=4)
    # fig.colorbar()
    plt.show()


def main1():
    import matplotlib.pyplot as plt

    all_wealth = []
    # This runs the model 100 times, each model executing 10 steps.
    for j in range(100):
        # Run the model
        model = MoneyModel(10)
        for i in range(10):
            model.step()

        # Store the results
        for agent in model.schedule.agents:
            all_wealth.append(agent.wealth)

    plt.hist(all_wealth, bins=range(max(all_wealth) + 1))

    plt.show()


def main2():
    model = MoneyModel(50, 10, 10)
    for i in range(20):
        model.step()

    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count
    plt.imshow(agent_counts, interpolation="nearest")
    plt.colorbar()
    plt.show()


def main3():
    model = MoneyModel(50, 10, 10)
    for i in range(100):
        model.step()

    gini = model.datacollector.get_model_vars_dataframe()
    gini.head()


if __name__ == '__main__':
    main4()
