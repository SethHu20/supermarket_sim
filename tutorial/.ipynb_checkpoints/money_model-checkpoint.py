from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector


def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.agents_limit
    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1 / N) - 2 * B



class MoneyAgent(Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 1

    def step(self):
        self.move()
        if self.wealth > 0:
            self.give_money()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1


class MoneyModel(Model):
    """A model with some number of agents."""

    def __init__(self, N, width, height):
        self.grid = MultiGrid(width, height, True)
        self.num_agents = N
        self.schedule = RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            model_reporters={"Gini": compute_gini}, agent_reporters={"Wealth": "wealth"}
        )

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()


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

    import matplotlib.pyplot as plt
    import numpy as np

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
    // main3()
    pass
