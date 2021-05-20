import random, copy
import numpy as np


# Should return representation of individual

def fps(pop) -> list:
    # If the fitness is negative, return fitness = 0
    # Make sure the fitness number works in positive numbers
    fitArray = np.array([i.fitness for i in pop.individuals])
    fitArray = fitArray/np.sum(fitArray.clip(min=0))
    fitArray = fitArray.clip(min=fitArray[fitArray > 0].min() / 2)
    fitArray /= np.sum(fitArray)

    if pop.optimization == "min":
        fitArray = (1 - fitArray/sum(fitArray))
        fitArray /= np.sum(fitArray)
    # Make a weighted choice
    choice = np.random.choice(pop.individuals, p=fitArray)
    # return the representation of the individual
    return choice.representation

def tournament(pop, size: int) -> list:
    """Tournament selection

    Args:
        pop (BasePopulation): Population
        size (int): size of tournament

    Returns:
        [list]: Individual representation
    """
    players = random.sample(pop.individuals, size)
    m = -1 if pop.optimization == "min" else 1
    winner = sorted(players, key= lambda i: i.fitness * m)[-1]
    # return the representation of the individual
    return winner.representation