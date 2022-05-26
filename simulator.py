from typing import Union

# import matplotlib.pyplot as plt
# import numpy as np
from matplotlib.colors import ListedColormap
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid, Coordinate
from mesa.time import RandomActivation, BaseScheduler


class WallAgent(Agent):

    def __init__(self, unique_id, model, product_type):
        super().__init__(unique_id, model)
        self.type = product_type


class CustomerAgent(Agent):

    def __init__(self, unique_id, model, demand_type):
        super().__init__(unique_id, model)
        self.demand = demand_type
        self.age = 0

    def step(self) -> None:
        self.model: AisleModel
        if self.demand != 0:
            for neighbour in self.model.grid.get_neighbors(self.pos, False, ):
                if isinstance(neighbour, WallAgent):
                    if neighbour.type == self.demand:
                        self.demand = 0
                        break
            x, y = self.pos
            x: int
            y: int
            if self.demand != 0:
                direction = self.model.direction_of_shelf(self.pos, self.demand)
                if self.model.grid.is_cell_empty((x, y - 1)):
                    self.model.grid.move_agent(self, (x, y - 1))
                elif self.model.grid.is_cell_empty((x + direction, y)):
                    self.model.grid.move_agent(self, (x + direction, y))
                else:
                    self.maybe_move_somewhere()
        elif self.demand == 0:
            if self.pos in self.model.exit:
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)
                self.model.datacollector.add_table_row("Lifespan", {
                    "unique_id": self.unique_id,
                    "age": self.age
                })
            else:
                x, y = self.pos
                x: int
                y: int
                if self.model.grid.is_cell_empty((x, y + 1)):
                    self.model.grid.move_agent(self, (x, y + 1))
                elif self.model.grid.is_cell_empty((x - 1, y)):
                    self.model.grid.move_agent(self, (x - 1, y))
                else:
                    self.maybe_move_somewhere()
        self.age += 1

    def maybe_move_somewhere(self):
        self.model: AisleModel
        if self.pos in self.model.exit:
            return
        self.model: AisleModel
        move_list = self.model.grid.get_neighborhood(self.pos, False, include_center=True)
        self.random.shuffle(move_list)
        for candidate in move_list:
            if self.model.grid.is_cell_empty(candidate):
                self.model.grid.move_agent(self, candidate)


class AisleModel(Model):

    def __init__(
            self,
            n: int = 100,
            width: int = 14,
            height: int = 4,
            shelf_config: Union[None, str, int] = None,
            g1_population: float = 0.334,
            g2_population: float = 0.333,
            g3_population: float = 0.333,
            spawn_probability: float = 0.3,
            seed: int = 0
    ):
        super().__init__()
        self.grid = SingleGrid(width, height, False)
        self.agents_limit = n
        self.total_agents = [0, 0, 0, 0]
        self.schedule = RandomActivation(self)
        self.running = True
        self.spawn_chance = spawn_probability

        self.probability_table = [g1_population, g2_population, g3_population]
        # if sum(self.probability_table) < 1.0:
        #     return Exception("Probability Table not proper")

        # wall creation
        if shelf_config is None:
            self.shelf_config = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]
        else:
            shelf_config = str(shelf_config)
            assert len(shelf_config) == 3
            self.shelf_config = [0] * (width - 10)
            for n in shelf_config:
                self.shelf_config.extend([int(n)] * 3)
            self.shelf_config.append(0)

        for i in range(width):
            # wall = WallAgent(None, self, 0)
            # self.grid.position_agent(wall, i, 0)
            wall = WallAgent(0, self, 0)
            self.grid.position_agent(wall, i, height - 1)

        for i, target_type in enumerate(self.shelf_config):
            wall = WallAgent(0, self, target_type)
            # self.grid.remove_agent(self.grid.get_cell_list_contents((i, 0)))
            self.grid.position_agent(wall, i, 0)

        for i in range(1, height - 1):
            wall = WallAgent(0, self, 0)
            self.grid.position_agent(wall, width - 1, i)

        self.entrance = [(0, 1)]
        self.exit = [(0, 2)] + self.entrance

        self.datacollector = DataCollector(
            model_reporters={
                "current_population": lambda model: model.schedule.get_agent_count(),
                "total_population": lambda x: x.total_agents[0],
                "G1_population": lambda x: x.total_agents[1],
                "G2_population": lambda x: x.total_agents[2],
                "G3_population": lambda x: x.total_agents[3],
            },
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
        if self.total_agents[0] < self.agents_limit and self.random.random() < self.spawn_chance:
            for potential_entrance in self.entrance:
                if self.grid.is_cell_empty(potential_entrance):
                    customer = CustomerAgent(self.next_id(), self, self.rand_type())
                    self.total_agents[0] += 1
                    self.total_agents[customer.demand] += 1
                    self.grid.position_agent(customer, *potential_entrance)
                    self.schedule.add(customer)
                    break
        if self.total_agents[0] >= self.agents_limit and self.schedule.get_agent_count() == 0:
            self.datacollector.collect(self)
            self.running = False


    def direction_of_shelf(self, pos: Coordinate, demand: int) -> int:
        shelf_left = self.shelf_config.index(demand)
        shelf_right = self.list_rindex(demand)
        x_coord = pos[0]
        if x_coord < shelf_left:
            return 1
        elif shelf_left <= x_coord <= shelf_right:
            return 1
        if shelf_right < x_coord:
            return -1

    def list_rindex(self, x):
        for i in reversed(range(len(self.shelf_config))):
            if self.shelf_config[i] == x:
                return i
        raise ValueError("{} is not in list".format(x))


# def main4():
#     model = AisleModel(10)
#
#     grid_point = np.zeros((model.grid.width, model.grid.height))
#
#     for cell, x, y in model.grid.coord_iter():
#         if isinstance(cell, WallAgent):
#             cell: WallAgent
#             grid_point[x][y] = cell.type
#         else:
#             grid_point[x][y] = -1
#
#     grid_point = grid_point.transpose()
#
#     fig, ax = plt.subplots()
#
#     ax.set_xticks(np.arange(model.grid.width), minor=False)
#     ax.set_yticks(np.arange(model.grid.height), minor=False)
#
#     cmap = ListedColormap(['white', 'black', 'red', 'green', 'blue'])
#
#     ax.imshow(grid_point, interpolation="nearest", origin='upper', extent=(0, model.grid.width, 0, model.grid.height),
#               cmap=cmap)
#     # psm = ax.pcolormesh(grid_point, rasterized=True, vmin=-4, vmax=4)
#     # fig.colorbar()
#     plt.show()
#
#
# def main1():
#     all_wealth = []
#     # This runs the model 100 times, each model executing 10 steps.
#     for j in range(100):
#         # Run the model
#         model = AisleModel(10)
#         for i in range(10):
#             model.step()
#
#         # Store the results
#         for agent in model.schedule.agents:
#             all_wealth.append(agent.unique_id)
#
#     plt.hist(all_wealth, bins=range(max(all_wealth) + 1))
#
#     plt.show()
#
#
# def main2():
#     model = AisleModel(50, 10, 10)
#     for i in range(20):
#         model.step()
#
#     agent_counts = np.zeros((model.grid.width, model.grid.height))
#     for cell in model.grid.coord_iter():
#         cell_content, x, y = cell
#         agent_count = len(cell_content)
#         agent_counts[x][y] = agent_count
#     plt.imshow(agent_counts, interpolation="nearest")
#     plt.colorbar()
#     plt.show()
#
#
# def main3():
#     model = AisleModel(50, 10, 10)
#     for i in range(100):
#         model.step()
#
#     gini = model.datacollector.get_model_vars_dataframe()
#     gini.head()
#
#
# if __name__ == '__main__':
#     main4()
