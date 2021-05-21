import random, copy
from ssl import OP_NO_TLSv1_1


def single_point_co(p1: list, p2: list):
    poopoo = random.randint(1, len(p1))
    offspring1 = p1[:poopoo] + p2[:poopoo]
    offspring2 = p1[poopoo:] + p2[poopoo:]

    return offspring1, offspring2


def cycle_co(p1: list, p2: list) -> tuple:
    # Randomly choose a parent to start the cycle
    if random.random() <= 0.5:
        p1, p2 = p2, p1
    r_len = len(p1)
    offspring1, offspring2 = {}, {}
    keys = offspring1.keys()
    idx = 0
    while len(keys) < r_len:
        if idx in keys:
            idx = min(list(set(range(r_len)) - set(keys)))
            p1, p2 = p2, p1
        else:
            offspring1[idx] = p1[idx]
            offspring2[idx] = p2[idx]
            idx = p2.index(p1[idx])
    print(offspring1)
    print(r_len)
    o1 = [offspring1[i] for i in range(r_len)]
    print(o1)
    o2 = [offspring2[i] for i in range(r_len)]
    return o1, o2

# TODO: Not working but has potential?
def cycle_co2(p1: list, p2: list) -> tuple:
    r_len = len(p1)
    offspring1, offspring2 = {}, {}
    keys = offspring1.keys()
    idx = random.choice(range(r_len))
    while len(keys) < r_len:
        if idx in keys:
            idx = random.choice(list(set(range(r_len)) - set(keys)))
            p1, p2 = p2, p1
        else:
            offspring1[idx] = p1[idx]
            offspring2[idx] = p2[idx]
            idx = p2.index(p1[idx])
    o1 = [offspring1[i] for i in range(r_len)]
    o2 = [offspring2[i] for i in range(r_len)]
    return o1, o2


def arithmetic_co(p1: list, p2: list) -> tuple:
    # set alpha
    a = random.uniform(0, 1)
    o1 = [p1[i] * a + (1 - a) * p2[i] for i, _ in enumerate(p1)]
    o2 = [p2[i] * a + (1 - a) * p1[i] for i, _ in enumerate(p1)]
    return o1, o2


def pmx_co(p1: list, p2: list) -> tuple:
    idx1, idx2 = sorted(random.sample(range(len(p1)), 2))

    def get_offspring(main_p, sec_p):
        o = main_p.copy()
        o_dict = {}
        # create window
        window_p1 = main_p[idx1:idx2]
        window_p2 = sec_p[idx1:idx2]
        mapping_p2 = {n: window_p1[i] for i, n in enumerate(window_p2)}
        o_dict = {i: sec_p[i] for i in range(idx1, idx2)}
        for i, n in enumerate(o):
            if i not in o_dict:
                if n not in window_p2:
                    o_dict[i] = n
                else:
                    m = mapping_p2[n]
                    while m in window_p2:
                        m = mapping_p2[m]
                    o_dict[i] = m

        return [o_dict[i] for i in sorted(o_dict)]

    o1 = get_offspring(p1, p2)
    o2 = get_offspring(p2, p1)
    return o1, o2


def crisp_co(p1: list, p2: list) -> tuple:
    idx1, idx2 = sorted(random.sample(range(len(p1)), 2))
    o1, o2 = p1, p2
    o1[idx1:idx2], o2[idx1:idx2] = p2[idx1:idx2], p1[idx1:idx2]
    return o1, o2


def gene_co(p1: list, p2: list, p_co: float) -> tuple:
    o1, o2 = p1, p2
    for i, _ in enumerate(o1):
        if random.random() <= p_co:
            o1[i] = p2[i]
            o2[i] = p1[i]
    return o1, o2


if __name__ == "__main__":
    mate1 = [3, 4, 8, 2, 7, 1, 6, 5]
    mate2 = [4, 2, 5, 1, 6, 8, 3, 7]

    print(cycle_co(mate1, mate2))
