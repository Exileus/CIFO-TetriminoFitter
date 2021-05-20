import random, copy
from library.main import Individual
import numpy as np

# TODO: I don't think it's storing the elites properly and carrying them to the next generation without mutations
def GAO(
    pop,
    generations: int,
    tournament_size: int,
    p_crossover: float = 0.1,
    p_mutation: float = 0.05,
    n_mutations: int = 1,
    p_rot_mut: float = 0.1,
    p_adoption: float = 0.01,
    verbose=False,
):
    m = -1 if pop.optimization == "min" else 1
    for _ in range(generations):
        for i in pop.individuals:
            pop.fitness(i)
        if verbose:
            print("ELITES:", pop.elites)
            print("ALL:", pop.individuals)
        pop.fitness_history.append(
            sorted(pop.individuals, key=lambda i: i.fitness * m, reverse=True)[
                0
            ].fitness
        )
        # Check if we want elitism and how many elites we keep
        if pop.n_elites > 0:
            pop_prime = copy.deepcopy(sorted(
                pop.individuals, key=lambda i: i.fitness * m, reverse=True
            ))[:pop.n_elites]
            if verbose:
                print("elite @ prime:", pop_prime)
        else:
            pop_prime = []
        # Evolve the rest of the population
        while len(pop_prime) < len(pop.individuals):
            # Select parents from population the representations of the parents
            i1, i2 = pop.selection(tournament_size), pop.selection(tournament_size)

            # Apply crossover ~ probability is given to function
            i1, i2 = pop.crossover(i1, i2, p_crossover)

            for i in [i1, i2]:
                # Apply mutation ~ probability is given to function
                i = pop.mutation(i, p_mutation, n_mutations)
                i = pop.orientation_mutation(i, p_rot_mut)
                # Create individual and get its fitness
                i = Individual(representation=i)
                pop.fitness(i)
                if len(pop_prime) < len(pop.individuals):
                    pop_prime.append(i)
            # Create a probability where parents can adopt a new
            if (len(pop_prime) < len(pop.individuals)) and (
                (adopted_i := pop.adoption(p_adoption)) != None
            ):
                pop.fitness(adopted_i)
                pop_prime.append(adopted_i)
        pop.individuals = copy.deepcopy(pop_prime)
        if verbose:
            print(sorted(pop.individuals, key=lambda i: i.fitness * m, reverse=True))
        pop.elites = copy.deepcopy(sorted(pop.individuals, key=lambda i: i.fitness * m, reverse=True)[:pop.n_elites])
        if verbose:
            print()
