from matplotlib.pyplot import grid
from library.hill_climbing.hc import hill_climb
import numpy as np
import random
from library.main import Individual, BasePopulation
import library.genetic_algorithm.selection as sel
import library.genetic_algorithm.mutation as mut
import library.genetic_algorithm.crossover as co
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances

T_DICT = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "L": [(0, 0), (1, 0), (2, 0), (2, 1)],
    "J": [(0, 0), (1, 0), (2, 0), (2, -1)],
    "T": [(0, 0), (1, 0), (1, -1), (1, 1)],
    "S": [(0, 0), (1, 0), (1, -1), (0, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (0, -1)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
}


def tetrimino_fitter(pieces: list, grid_shape: tuple, verbose: bool = False) -> tuple:
    """Fits all pieces into a grid with a given shape.

    Args:
        pieces (list): List with piece type and rotation as str
        grid_shape (tuple): (x, y)
        verbose (bool, optional): Defaults to False.

    Returns:
        (tuple): Filled grid, pieces' coordinates
    """
    # Create pieces_coordinates list
    pieces_coordinates = []

    # Initialize grid
    grid = np.zeros(grid_shape)
    # Transform the corresponding piece into its matrix representation.
    # Consider rotation and how that impacts the piece, and apply the corresponding transformation here.
    for each in pieces:
        piece = get_piece_coordinates(each)

        # Identify spots that can be filled.
        free_space_array = np.column_stack(
            np.where(grid == 0)
        )  # this can be used to find the coordinates of all the zeros, and have them stored.
        for coord in free_space_array:
            if verbose:
                print(f"First zero in sequence found at {tuple(coord)}")

            # get array for piece and displace it by the coordinates
            p = piece + coord

            # check that piece is not out of bounds
            if not (
                (-1 in p)
                or (np.max(p[:, 0]) > (grid.shape[0] - 1))
                or (np.max(p[:, 1]) > (grid.shape[1] - 1))
            ):
                # Get the test fit
                test_fit = np.array([grid[x, y] for x, y in p])
                # If the sum is 0, it fits
                if np.sum(test_fit) == 0:
                    # Fill up the spaces with 1
                    for x, y in p:
                        grid[x, y] = 1
                    # Add the piece's coordinates to the list
                    pieces_coordinates.append(p)
                    # We placed the piece so we stop trying to fit the piece
                    break
    return grid, pieces_coordinates


def get_piece_coordinates(piece: str):

    piece_type = piece[0]
    c = np.array(T_DICT[piece_type])
    rotation = int(piece[1])

    if rotation == 1:
        c = np.flip(c, axis=1)
        c[:, 1] *= -1
    elif rotation == 2:
        c *= -1
    elif rotation == 3:
        c = np.flip(c, axis=1)
        c[:, 0] *= -1
    return c


def pieces_generator(grid_shape: tuple, rotation=True):
    # list of all pieces
    pieces_types = perfect_block4(grid_shape)
    random.shuffle(pieces_types)
    if rotation:
        for i in range(len(pieces_types)):
            piece_rotation = str(random.randint(1, 4))
            pieces_types[i] += piece_rotation


    return pieces_types


def perfect_block4(grid_shape: tuple):
    """Select random blocks 4x4 with a perfect pieces match.

    Args:
        grid_shape (tuple): (x, y)
    Returns:
        list: Pieces types to be used
    """
    assert (grid_shape[0] % 4 == 0) & (grid_shape[1] % 4 == 0), "grid shapes must be multiples of 4"
    n_blocks = grid_shape[0]//4 * grid_shape[1]//4
    possible_blocks = [["O", "O", "O", "O"],
                       ["I", "J", "T", "T"],
                       ["Z", "Z", "L", "L"],
                       ["T", "T", "Z", "L"],
                       ["I", "I", "I", "I"],
                       ["I", "I", "O", "O"],
                       ["O", "L", "J", "I"],
                       ["T", "T", "T", "T"],
                       ["S", "L", "L", "I"],
                       ["Z", "L", "J", "I"],
                       ["S", "S", "J", "J"],
                       ["S", "T", "T", "J"]]
    return [piece for sublist in random.choices(possible_blocks, k=n_blocks) for piece in sublist]


class Population(BasePopulation):
    def __init__(
        self,
        individuals: list,
        optimization: str,
        n_elites: int,
        valid_set: list,
        grid_shape: tuple,
    ):
        super().__init__(
            individuals, optimization, n_elites=n_elites, valid_set=valid_set
        )
        self.grid_shape = grid_shape

    def fitness(self, individual: Individual):
        """Calculate individual fitness and update the individual's fitness.

        Args:
            individual (Individual): initiated individual class
        """
        # simplify code
        rep = individual.representation
        # calculate fitness
        grid, _ = tetrimino_fitter(rep, self.grid_shape)
        occupation_fitness = np.sum(grid) * 100
        compactness_fitness = 0
        if 0 in grid:
            empties = np.column_stack(np.where(grid == 0))
            empties = manhattan_distances(empties)
            # The closer the empty spaces are, the smaller the distances.
            compactness_fitness = np.sum(empties)
        # update individual's fitness
        individual.fitness = occupation_fitness - compactness_fitness

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

    def selection_tournament(self, *args, **kwargs):
        return sel.tournament(self, *args, **kwargs)

    def selection_fps(self, *args, **kwargs):
        return sel.fps(self, *args, **kwargs)

    def orientation_mutation(self, representation: list, p_mutation: float) -> list:
        for i, (letter, number) in enumerate(representation):
            if random.random() <= p_mutation:
                valid_set_c = [0, 1, 2, 3]
                valid_set_c.remove(int(number))
                representation[i] = letter + str(random.choice(valid_set_c))
        return representation

    def mutation_swap(self, representation, p_mutation, n_mutations) -> list:
        representation = mut.swap_mutation(representation, p_mutation, n_mutations)
        return representation

    def mutation_inverted(self, representation, p_mutation) -> list:
        representation = mut.inversion_mutation(representation, p_mutation)
        return representation

    def crossover(
        self,
        p1: Individual,
        p2: Individual,
        p_crossover: float,
        crossover_type: "cycle or pmx" = "pmx",
    ):
        """Crossover algorithm. Either applies cycle crossover or partially matched crossover.

        Args:
            p1 (Individual): parent 1
            p2 (Individual): parent 2
            p_crossover (float): probability of crossing over
            crossover_type (cycle or pmx, optional): string cycle or pmx. Defaults to "pmx".

        Returns:
            o1, o2: offspring1, offspring2 (or parents if no crossover)
        """

        if random.random() < p_crossover:
            # Creating index system
            p1_dict = {i: l + str(n) for i, (l, n) in enumerate(p1)}
            tmp = p1_dict.copy()
            p2_dict = {}
            for i, (l, n) in enumerate(p2):
                for key, value in tmp.items():
                    if value[0] == l:
                        p2_dict[key] = l + str(n)
                        del tmp[key]
                        break
            p1_encoded = list(p1_dict.keys())
            p2_encoded = list(p2_dict.keys())
            # run crossover
            if crossover_type == "cycle":
                o1, o2 = co.cycle_co(p1_encoded, p2_encoded)
            else:
                o1, o2 = co.pmx_co(p1_encoded, p2_encoded)
            # Recreate from index
            for i, o in enumerate(o1):
                if o == p1_encoded[i]:
                    o1[i] = p1_dict[o]
                else:
                    o1[i] = p2_dict[o]
            for i, o in enumerate(o2):
                if o == p2_encoded[i]:
                    o2[i] = p2_dict[o]
                else:
                    o2[i] = p1_dict[o]
            return o1, o2
        else:
            return p1, p2

    def adoption(self, p_adoption: float, hc_hardstop: int = 0) -> Individual:
        """How likely some parents can adopt an Individual

        Args:
            p_adoption (float): Probability of parents adopting an individual

        Returns:
            Individual
        """
        if random.random() <= p_adoption:
            ind = generate_individual(
                self.valid_set, self.grid_shape, hc_hardstop=hc_hardstop
            )
            return ind


def generate_individual(
    valid_list: list, grid_shape: tuple, hc_hardstop: int = 0, hc=False, 
) -> Individual:
    ind = [
        Individual(
            [
                letter + str(random.randint(0, 3))
                for letter in random.sample(valid_list, len(valid_list))
            ]
        )
    ]
    if hc:
        pop = Population(
            ind, "max", n_elites=1, valid_set=valid_list, grid_shape=grid_shape
        )
        hill_climb(pop, hardstop=hc_hardstop)
        return pop.elites[0]
    else:
        return ind[0]
