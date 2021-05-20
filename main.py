import numpy as np
import random
from library.main import Individual, BasePopulation
import library.genetic_algorithm.selection as sel
import library.genetic_algorithm.mutation as mut
import library.genetic_algorithm.crossover as co
from sklearn.metrics.pairwise import manhattan_distances

T_DICT = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "L": [(0, 0), (1, 0), (2, 0), (2, 1)],
    "J": [(0, 0), (1, 0), (2, 0), (2, -1)],
    "T": [(0, 0), (1, 0), (1, -1), (1, 1)],
    "S": [(0, 0), (1, 0), (1, -1), (0, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (0, -1)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
}


def tetrimino_fitter(pieces: list, grid_shape: tuple, verbose: bool = False):
    """Fits all pieces into the grid.

    Args:
        pieces (str): piece to fit
        grid (np.array, optional): Grid expects an array. Defaults to None.
        pieces_coordinates (list, optional): [description]. Defaults to [].
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        grid, pieces_coordinates: Returns the grid with the "fitted" pieces and list with piece coordinates.
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
                or (max(p[:, 0]) > (grid.shape[0] - 1))
                or (max(p[:, 1]) > (grid.shape[1] - 1))
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
    n_filled = 0
    max_fitness = grid_shape[0]* grid_shape[1]
    pieces = []

    while n_filled < max_fitness:
        piece_type = random.choice(list(T_DICT.keys()))
        if rotation:
            piece_rotation = str(random.randint(1, 4))
            pieces.append(piece_type + piece_rotation)
        else:
            pieces.append(piece_type)
        n_filled += 4
    return pieces


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

        empties = np.column_stack(
            np.where(grid == 0)
        ) 

        # The closer the empty spaces are, the smaller the distances.
        apartness_fit = np.sum(manhattan_distances(empties))
        # update individual's fitness
        individual.fitness = np.sum(grid) * 100 - apartness_fit

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

    def selection(self, *args, **kwargs):
        return sel.tournament(self, *args, **kwargs)

    def orientation_mutation(self, representation: list, p_mutation: float) -> list:
        for i, (letter, number) in enumerate(representation):
            if random.random() <= p_mutation:
                valid_set_c = [0, 1, 2, 3]
                valid_set_c.remove(int(number))
                representation[i] = letter + str(random.choice(valid_set_c))
        return representation

    def mutation(self, representation, p_mutation, n_mutations) -> list:
        representation = mut.swap_mutation(representation, p_mutation, n_mutations)
        # representation = self.orientation_mutation(representation, p_mutation)
        return representation

    def crossover(self, p1, p2, p_crossover: float):
        if random.random() < p_crossover:
            p1_dict = {i: l+str(n) for i, (l, n) in enumerate(p1)}
            tmp = p1_dict.copy()
            p2_dict = {}
            for i, (l, n) in enumerate(p2):
                for key, value in tmp.items():
                    if value[0] == l:
                        p2_dict[key] = l+str(n)
                        del tmp[key]
                        break
            p1_encoded = list(p1_dict.keys())
            p2_encoded = list(p2_dict.keys())
            o1, o2 = co.pmx_co(p1_encoded, p2_encoded)
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

    def adoption(self, p_adoption: float) -> Individual:
        """How likely some parents can adopt an Individual

        Args:
            p_adoption (float): Probability of parents adopting an individual

        Returns:
            Individual
        """
        if random.random() <= p_adoption:
            return Individual([piece+str(random.randint(0, 3)) for piece in random.sample(self.valid_set, len(self.valid_set))])

def generate_individual(valid_list):
    return Individual([letter+str(random.randint(0, 3)) for letter in random.sample(valid_list, len(valid_list))])