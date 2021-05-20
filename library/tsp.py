from main import Individual, BasePopulation
from tsp_data import distance_matrix
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
        fitness = [distance_matrix[rep[i - 1]][rep[i]] for i, _ in enumerate(rep)]
        # update individual's fitness
        individual.fitness = sum(fitness)

    def neighbours(self, individual: Individual):
        """Generate neighbors of individual by swapping each node pair.

        Args:
            individual (Individual): individual to which we update the neighbors
    """
        rep = individual.representation
        neighbours = [rep.copy() for _ in enumerate(rep[:-1])]
        for i, indv in enumerate(neighbours):
            indv[i], indv[i + 1] = indv[i + 1], indv[i]
        individual.neighbours = [Individual(n) for n in neighbours]
        
if __name__ == "__main__":
    pass
