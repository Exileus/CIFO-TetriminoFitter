from logging import raiseExceptions
import random, copy
from library.main import Individual
from tetris_main import Population
import numpy as np


def GAO(
    pop: Population,
    generations: int,
    selection_type: "tournament/fps" = "tournament",
    tournament_size: int = 2,
    crossover_type: "cycle/pmx" = "pmx",
    p_crossover: float = 0.1,
    mutation_type: "swap/inverted" = "swap",
    p_mutation: float = 0.05,
    n_mutations: int = 1,
    p_rot_mut: float = 0.1,
    p_adoption: float = 0.01,
    hc_hardstop: int = 0,
    verbose=False,
):
    m = -1 if pop.optimization == "min" else 1
    for _ in range(generations):
        for i in pop.individuals:
            pop.fitness(i)
        pop.fitness_history.append(
            sorted(pop.individuals, key=lambda i: i.fitness * m, reverse=True)[
                0
            ].fitness
        )
        # Check if we want elitism and how many elites we keep
        if pop.n_elites > 0:
            pop_prime = copy.deepcopy(
                sorted(pop.individuals, key=lambda i: i.fitness * m, reverse=True)
            )[: pop.n_elites]
        else:
            pop_prime = []
        # Evolve the rest of the population
        while len(pop_prime) < len(pop.individuals):
            # Select parents from population the representations of the parents
            if selection_type == "tournament":
                i1, i2 = (
                    pop.selection_tournament(tournament_size),
                    pop.selection_tournament(tournament_size),
                )
            else:
                i1, i2 = pop.selection_fps(), pop.selection_fps()
            # Apply crossover ~ probability is given to function
            i1, i2 = pop.crossover(i1, i2, p_crossover, crossover_type)

            for i in [i1, i2]:
                # Apply mutation ~ probability is given to function
                if mutation_type == "swap":
                    i = pop.mutation_swap(i, p_mutation, n_mutations)
                elif mutation_type == "inverted":
                    i = pop.mutation_inverted(i, p_mutation)
                i = pop.orientation_mutation(i, p_rot_mut)
                # Create individual and get its fitness
                i = Individual(representation=i)
                pop.fitness(i)
                if len(pop_prime) < len(pop.individuals):
                    pop_prime.append(copy.deepcopy(i))
            # Create a probability where parents can adopt a new
            if (len(pop_prime) < len(pop.individuals)) and (
                (adopted_i := pop.adoption(p_adoption, hc_hardstop=hc_hardstop)) != None
            ):
                pop.fitness(adopted_i)
                pop_prime.append(adopted_i)
        pop.individuals = copy.deepcopy(pop_prime)

        # Sort all individuals by fitness
        elites = sorted(pop.individuals, key=lambda i: i.fitness * m, reverse=True)
        # if we're using elitism, keep the best N elites
        if pop.n_elites > 0:
            pop.elites = copy.deepcopy(elites[: pop.n_elites])

        # keep the best individual of population
        else:
            pop.elites = copy.deepcopy(elites[0])
        if verbose:
            print(f"GEN {_} ELITES: {pop.elites}")
