from typing import Any
import numpy as np


class BasePopulation:
    """Population for optimization
    """

    def __init__(
        self,
        individuals: list,
        optimization: str,
        n_elites: int = 0,
        valid_set: list = None,
    ):
        """Args:
            individuals (list): List of Individual
            optimization (str): max or min
            n_elites (int, optional): Number of individuals to retain at each generation. Defaults to 0.

        Raises:
            ValueError: [description]
        """
        self.individuals = individuals
        self.optimization = optimization.lower()
        self.n_elites = n_elites
        self.valid_set = valid_set
        self.fitness_history = []
        self.elites = []
        if self.optimization not in ["min", "max"]:
            raise ValueError("Set optimization to min or max.")

    def fitness(self):
        pass

    def neighbours(self):
        pass

    def selection(self):
        pass

    def crossover(self):
        pass

    def mutation(self):
        pass


class Individual:
    def __init__(self, representation: list) -> None:
        self.representation = representation
        self.fitness: Any = None
        self.neighbours = []

    def __len__(self):
        return len(self.representation)

    def __repr__(self):
        return f"Individual <{self.fitness}>"
