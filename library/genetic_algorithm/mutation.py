import random

def gene_mutation(representation: list, p_mutation: float, valid_set: list) -> list:
    for i, gene in enumerate(representation):
        if random.random() <= p_mutation:
            valid_set_c = valid_set.copy()
            valid_set_c.remove(gene)
            representation[i] = random.choice(valid_set_c)
    return representation


def single_mutation(representation: list, p_mutation: float, valid_set: list) -> list:
    if random.random() <= p_mutation:
        # get random point in representation
        idx = random.randint(0, len(representation) - 1)
        # copy valid set of values and make sure we don't get the same value
        valid_set_c = valid_set.copy()
        valid_set_c.remove(representation[idx])

        # change the representation point
        representation[idx] = random.choice(valid_set_c)
    return representation

def swap_mutation(representation: list, p_mutation: float, n_mutations: int) -> list:
    r = representation
    for _ in range(n_mutations):
        if random.random() <= p_mutation:
            idx1, idx2 = random.sample(range(len(r)), 2)
            r[idx1], r[idx2] = r[idx2], r[idx1]
    return r

def inversion_mutation(representation: list, p_mutation: float) -> list:
    r = representation
    if random.random() <= p_mutation:
        idx1, idx2 = sorted(random.sample(range(len(r)), 2))
        r[idx1:idx2] = r[idx1:idx2][::-1]
    return r