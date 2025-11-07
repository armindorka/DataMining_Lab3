import numpy as np


def elastic(
    x,
    y,
    *,
    init_border,
    compute_cost,
    canonical=0,
    alternate_row=0,
    alternate_column=0,
):
    """
    Generic function for computing the elastic distance between `x` and `y`

    Parameters
    ----------
    x : ndarray of shape (m, )
        The first array.
    y : ndarray of shape (n, )
        The second array.
    init_border : function
        Function to initialize the border f(x, i, y, j) -> float
    compute_cost : function
        Function to compute the cost between point i and j f(x, i, y, j) -> float
    canonical : float
        Cost added to the diagonal.
    alternate_row : float
        Cost added to the previous row.
    alternate_column : float
        Cost added to the previous column.

    Returns
    -------
    float
        The elastic distance.
    """
    m, n = len(x), len(y)
    cost = np.zeros(max(m, n))
    cost_prev = np.zeros(max(m, n))

    cost_prev[0] = init_border(x, 0, y, 0)
    for i in range(1, min(m, n)):
        cost_prev[i] = cost_prev[i - 1] + init_border(x, 0, y, i)

    for i in range(1, m):
        cost[0] = cost_prev[0] + init_border(x, i, y, 0)
        for j in range(1, n):
            cost[j] = min(
                cost_prev[j - 1] + canonical,
                cost_prev[j] + alternate_row,
                cost[j - 1] + alternate_column,
            ) + compute_cost(x, i, y, j)
        cost, cost_prev = cost_prev, cost  # SWAP

    return cost_prev[n - 1]


def init_border_dtw(x, i, y, j):
    return (x[i] - y[j])**2 
 

def compute_cost_dtw(x, i, y, j):
    # TODO: euclidean distance between points
    return (x[i] - y[j])**2


class WDTW:
    def __init__(self, m, n, g=0.5):
        # Weight array initialized according to Jeong et. al. (2007)
        self.W = np.array(
            [1 / (1 + np.exp(-g * (i - max(m, n) / 2))) for i in range(max(m, n))]
        )

    def init_border(self, x, i, y, j):
        # TODO euclidean distance between points with weight
         return (x[i] - y[j])**2 * self.W[abs(i-j)]

    def compute_cost(self, x, i, y, j):
        # TODO euclidean distance between points with weight
         return (x[i] - y[j])**2 * self.W[abs(i- j)]
