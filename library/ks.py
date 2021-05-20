from main import Individual, BasePopulation
from ks_data import weights, values, capacity
import numpy as np


class Population(BasePopulation):
    def fitness(self, individual: Individual):
        """Calculate individual fitness and update the individual's fitness.

        Args:
            individual (Individual): initiated individual class
        """
        # simplify code
        rep = individual.representation
        # calculate fitness
        individual.weight = sum(rep * np.array(weights))
        if individual.weight > capacity:
            individual.fitness = capacity - individual.weight
        else:
            individual.fitness = sum(rep * np.array(values))

    def neighbours(self, individual: Individual):
        """Generate neighbors of individual by swapping one bit at a time in each position

        Args:
            individual (Individual): individual to which we update the neighbors
        """
        rep = individual.representation
        neighbours = [rep.copy() for _ in enumerate(rep[:-1])]
        for i, ind in enumerate(neighbours):
            ind[i] ^= 1
        individual.neighbours = [Individual(n) for n in neighbours]


if __name__ == "__main__":
    pass
