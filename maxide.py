import numpy as np
import numba as nb
from time import time
import scipy as sp

@nb.njit(nb.double[:, :](nb.double[:, :], nb.double[:, :], nb.int64[:], nb.int64[:]))
def partial_matrix_mult(first_matrix, second_matrix, coordinate_rows, coordinate_columns):
    """Partial Matrix Multiplication according to an input list of coordinates for the sparse matrix. Replaces xumm.c

    Args:
        first_matrix (float64[:,:]): First matrix to be multiplied
        second_matrix (float64[:,:]): Second matrix to be multiplied
        coordinate_rows (int64[:]): List of the coordinates for rows where entries of pre-defined matrix are non-zero
        coordinate_columns (int64[:]): List of coordinates for columns where entries of pre-defined matrix are non-zero

    Returns:
        float64[:,:]: Resulting partial matrix multiplication of both inputs
    """
    first_matrix_rows     =  first_matrix.shape[0]
    second_matrix_columns = second_matrix.shape[1]
    
    return_matrix = np.zeros((first_matrix_rows, second_matrix_columns))

    for nonzero_index in range(len(coordinate_rows)):
        row_number = coordinate_rows[nonzero_index]
        col_number = coordinate_columns[nonzero_index]

        return_matrix[row_number, col_number] = np.dot(first_matrix[row_number], second_matrix[:, col_number])

    return return_matrix

def sides_svd_2_threshold(input_matrix_1, input_matrix_2, input_matrix_3, L, learning_rate):
    """Thresholding function for the SVD (I think)

    Args:
        input_matrix_1 (float64[:,:]): Left   input matrix
        input_matrix_2 (float64[:,:]): Middle input matrix
        input_matrix_3 (float64[:,:]): Right  input matrix
        L (float64): L, whatever that means. TODO: RENAME THIS
        learning_rate (float64): I THINK this is the learning rate. TODO: RENAME THIS

    Returns:
        float64[:,:]: thresholded SVD
    """
    interim_matrix_A = input_matrix_1 - input_matrix_2 / L + input_matrix_3 / L
    [svd_matrix_L, svd_matrix_S, svd_matrix_T] = np.linalg.svd(interim_matrix_A)
    #Turn all negatives to zero
    svd_matrix_S = np.clip(svd_matrix_S - learning_rate / L, a_min = 0, a_max = None)
    return svd_matrix_L @ svd_matrix_S @ svd_matrix_T.T

def maxide(input_matrix: np.ndarray, side_matrix_A: np.ndarray, side_matrix_B: np.ndarray, regularization_param: float, max_iterations: int = 100):

    time_start = time()

    #Basic definitions
    [input_size_n, input_size_m] = input_matrix.shape
    

    #Generate an array that contains all the row values and all the column values for the non-zeroes in the input matrix
    input_matrix_non_zero_mask              = input_matrix > 0
    [nonzero_row_array, nonzero_col_array]  = np.argwhere(input_matrix_non_zero_mask).T
    nonzero_col_array = nonzero_col_array.T

    #Determine the number of columns for each of the side matrices
    matrix_A_size_ra = side_matrix_A.shape[1]
    matrix_B_size_rb = side_matrix_B.shape[1]

    #TODO: RENAME THESE WHEN IT'S DISCOVERED WHAT THEY DO
    L                            = 1
    gamma                        = 2
    completion_matrix_initial_Z0 = np.zeros([matrix_A_size_ra, matrix_B_size_rb])
    completion_matrix_Z          = completion_matrix_initial_Z0.copy()
    alpha_0                      = 1
    alpha                        = 1

    convergence_matrix = np.zeros([max_iterations, 1])

    #Determine whether the inputs are for a multi-label problem
    multi_label_bool = np.array_equal(side_matrix_B, np.eye(input_size_m))
    
    #Since multi-label problems have the side information matrix B as the identity, we can save a lot of computation by
    #not multiplying the product of side matrix A and the input matrix with side matrix B, since the result will be unchanged
    if multi_label_bool:
        stvdt3 = side_matrix_A.T @ input_matrix
        matrix_A_Z0_B_Omega = np.zeros(side_matrix_A.shape[0], matrix_B_size_rb)[input_matrix_non_zero_mask]
    else:
        stvdt3 = side_matrix_A.T @ input_matrix @ side_matrix_B
        matrix_A_Z0_B_Omega = np.zeros(side_matrix_A.shape[0], side_matrix_B.shape[0])[input_matrix_non_zero_mask]

    matrix_A_Z_B_Omega = matrix_A_Z0_B_Omega.copy()

    #Matrix Completion iteration
    for iteration in range(max_iterations):
        alpha_constant = alpha * (1 / alpha_0 - 1)
        #Update completion matrix Z
        interim_matrix_Y = completion_matrix_Z + alpha_constant * (completion_matrix_Z - completion_matrix_initial_Z0)
        completion_matrix_initial_Z0 = completion_matrix_Z.copy()

        matrix_A_Y_B_Omega = (1 + alpha_constant) * matrix_A_Z_B_Omega - alpha_constant * matrix_A_Z0_B_Omega
        
        #Create a sparse version to speed up matrix multiplication
        sparse_A_Y_B_Omega = sp.sparse.coo_matrix((matrix_A_Y_B_Omega, nonzero_row_array, nonzero_col_array.T), shape=(input_size_n, input_size_m))
        
        if multi_label_bool:
            stvdt2 = side_matrix_A.T @ sparse_A_Y_B_Omega
        else:
            stvdt2 = side_matrix_A.T @ sparse_A_Y_B_Omega @ side_matrix_B
        
        