import numpy as np
import random
from library.main import Individual, BasePopulation

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


def pieces_generator(grid_shape: tuple):
    n_filled = 0
    max_fitness = grid_shape[0]* grid_shape[1]
    pieces = []

    while n_filled < max_fitness:
        piece_type = random.choice(list(T_DICT.keys()))
        piece_rotation = str(random.randint(1, 4))
        pieces.append(piece_type + piece_rotation)
        n_filled += 4
    return pieces


class Population(BasePopulation):
    def __init__(
        self,
        individuals: list,
        optimization: str,
        n_elites: int,
        valid_set: list,
        grid_shape: int,
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
        f, _ = tetrimino_fitter(rep, self.grid_shape)
        # update individual's fitness
        individual.fitness = np.sum(f)

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
