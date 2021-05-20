import random, copy
from main import BasePopulation, Individual
import numpy as np


def hill_climb(
    pop: BasePopulation,
    verbose: bool = False,
    hardstop: int = 100,
) -> Individual:
    """Hill Climb on a population of individuals by giving the function and
    fitness function and a function to get neighbours. Return the best individual.

    Args:
        pop (BasePopulation): Population to hill climb

    Returns:
        Individual: Return the best individual found
    """

    # loop through each individual in the population
    for i, ind in enumerate(pop.individuals):

        # invert fitness if minimizing
        m = -1 if pop.optimization == "min" else 1
        visited = {}
        hardstop_counter = 0
        while True:
            pop.fitness(ind)
            if verbose:
                print()
            if verbose:
                print(f"{ind.fitness = }")
            ind_representation_str = "".join([str(i) for i in ind.representation])
            if ind_representation_str in visited.keys():
                if verbose:
                    print(f"{ind_representation_str = } found in visited individuals")
                ind = copy.deepcopy(visited[ind_representation_str])
                ind.number_n_visited += 1
                if ind.number_n_visited == ind.len_max:
                    if verbose:
                        print("maximum neighbours visited in plateau")
                    # return ind
                    pop.individuals[i] = copy.deepcopy(ind)
                    break
                ns_fitness = (
                    [n.fitness for n in ind.neighbours] * np.array(m)
                ).tolist()
                if verbose:
                    print(f"{ind.len_max = }, {ind.number_n_visited = }")
            else:
                if verbose:
                    print(f"{ind_representation_str = }")
                pop.neighbours(ind)
                for n in ind.neighbours:
                    pop.fitness(n)
                # Calculate fitness of neighbours given optimization type by m
                ns_fitness = (
                    [n.fitness for n in ind.neighbours] * np.array(m)
                ).tolist()
                # How many neighbours have the best fitness
                ind.len_max = sum((np.array(ns_fitness) == max(ns_fitness)))
                ind.number_n_visited = 0
                if verbose:
                    print(f"{ind.len_max = }, {ind.number_n_visited = }")

            # Get best neighbour
            best_n_idx = ns_fitness.index(max(ns_fitness))
            best_n = ind.neighbours[best_n_idx]
            if verbose:
                print(
                    f"{best_n.fitness = } ;; {ind.fitness = } ;; diff = {ind.fitness - best_n.fitness}"
                )
            if (best_n.fitness * m) > (ind.fitness * m):
                if verbose:
                    print(f"{(best_n.fitness * m) = } > {(ind.fitness * m) = }")
                visited = {}
                ind = copy.deepcopy(best_n)

            elif best_n.fitness == ind.fitness:
                if hardstop_counter == hardstop:
                    pop.individuals[i] = copy.deepcopy(ind)
                    break
                else:
                    hardstop_counter += 1
                if verbose:
                    print(f"Found plateau on {best_n.representation} and {ind.representation}")
                # Change the best neighbour's fitness value to never be max/min again
                ind.neighbours[best_n_idx].fitness -= 1 * m
                visited[ind_representation_str] = copy.deepcopy(ind)
                ind = copy.deepcopy(best_n)
            else:
                if verbose:
                    print(f"No better individuals found.")
                pop.individuals[i] = copy.deepcopy(ind)
                break
    if pop.n_elites > 0:
        pop.elites = sorted(pop.individuals, key= lambda i: i.fitness)[:pop.n_elites]