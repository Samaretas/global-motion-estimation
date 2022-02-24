import multiprocessing
from xmlrpc.client import MAXINT
import numpy as np
from utils import timer
from utils import get_pyramids
from temp_bbme import Block_matcher
from diy_bbme import get_motion_fied


N_MAX_ITERATIONS = 50
DIRECT_INVERSE = True

np.seterr(all='raise')

def sum_squared_differences(previous, current):
    """
    Computes the sum of squared differences between frames.

    Args:
        previous (np.ndarray):  previous frame
        current (np.ndarray):  current frame
    """
    if type(previous) != np.ndarray or type(current) != np.ndarray:
        raise Exception("Should use only numpy arrays as parameters")
    if previous.shape != current.shape:
        raise Exception("Frames must have the same shape")
    E = current - previous
    E = E * E
    return E


def dense_motion_estimation(previous, current):
    """
    Given a couple of frames, estimates the dense motion field.

    The dense motion field corresponds to a matrix of size the # of blocks
    that fit in the image for each dimension. Each element of the matrix
    contains two values, the shift in x and y directions.

    Args:
        previous (np.ndarray): previous frame of the video
        current (np.ndarray): current frame of the video

    Returns:
        motion_field (np.ndarray): estimated dense motion field
    """
    motion_field = get_motion_fied(
        previous, current, block_size=6, searching_procedure=3, search_window=2
    )  # diamond search

    # BM = Block_matcher(block_size=6, search_range=2, pixel_acc=1, searching_procedure=2)
    # _, motion_field = BM.get_motion_field(previous, current)

    return motion_field


def compute_first_parameters(dense_motion_field):
    """
    Given the initial motion field, returns the first estimation of the parameters.

    Args:
        dense_motion_field (np.ndarray): ndarray with shape[-1]=2; the touples in the last dimension represent the shift of the pixel from previous to current frame. It is computed via dense motion field estimation.

    Returns:
        list[float]: the list of parameters of the motion model
    """
    a0 = np.mean(dense_motion_field[:, :, 0])
    a1 = np.mean(dense_motion_field[:, :, 1])
    #! first initialization needs a2 and a5 to be 1, otherwise the prediction will be completely blank
    (a2, a3, a4, a5, a6, a7) = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    return [a0, a1, a2, a3, a4, a5, a6, a7]


def motion_model(p, x, y):
    """
    Given the current parameters and the coordinates of a point, computes the compensated coordinates.
    Note that since the motion model is separated from the rest of the code, it can be easily changed with others; this means that trying different types of motion model should be as easy as writing them in this function.
    Note: w/ this you should chanfe also compute_first_parameters.

    Args:
        parameters (list): parameters of the motion model
        x (int): x coordinate to translate with the motion model
        y (int): y coordinate to translate with the motion model

    Returns:
        tuple[int,int]:     couple of coordinates, both translated w/ motion model
    """
    # TODO: change this with a more conscious approximation (<> x.5)
    try:
        x1 = int((p[0] + p[2] * x + p[3] * y) / (p[6] * x + p[7] * y + 1))
        y1 = int((p[1] + p[4] * x + p[5] * y) / (p[6] * x + p[7] * y + 1))
    except:
        print(f"Denominator problem")
        # print(f"Denominator problem: {(p[6]*x+p[7]*y+1)} cannot be used")
        # print(f"parameters p[6]:{p[6]} p[7]:{p[7]} x:{x} y:{y}")
        x1 = y1 = MAXINT
    return (x1, y1)


def compute_compensated_frame_complete(previous: np.ndarray, parameters: list):
    """
    Computes I' given I and the current parameters. Differently from the non-corrected version, it also corrects noise with the use of a small convolutional kernel.

    Args:
        previous (np.ndarray): previous frame
        parameters (list): current parameters

    Returns:
        np.ndarray: motion-compensated frame
    """
    compensated_coordinates = []
    for i in range(previous.shape[0]):
        for j in range(previous.shape[1]):
            x1, y1 = motion_model(parameters, i, j)
            compensated_coordinates.append([x1, y1])
    # sanitize the coordinates (respect for thr boundaries)
    for i in range(len(compensated_coordinates)):
        if compensated_coordinates[i][0] < 0:
            compensated_coordinates[i][0] = 0
        elif compensated_coordinates[i][0] >= previous.shape[0]:
            compensated_coordinates[i][0] = previous.shape[0] - 1

        if compensated_coordinates[i][1] < 0:
            compensated_coordinates[i][1] = 0
        elif compensated_coordinates[i][1] >= previous.shape[1]:
            compensated_coordinates[i][1] = previous.shape[1] - 1
    # create compensated frame with the compensated coordinates
    compensated_image = np.copy(previous)
    mask = np.zeros_like(previous, dtype=bool)
    for i in range(previous.shape[0]):
        for j in range(previous.shape[1]):
            cc = compensated_coordinates.pop(0)  # compensated_coordinate
            compensated_image[cc[0]][cc[1]] = previous[i][j]
            mask[cc[0]][cc[1]] = True
    # the mask identifies the points where we find noise, namely the points that retain the value of the previous frame
    for i in range(1, previous.shape[0] - 1):
        for j in range(1, previous.shape[1] - 1):
            if not mask[i][j]:
                compensated_image[i][j] = int(
                    (
                        np.sum((compensated_image[i - 1 : i + 2, j - 1 : j + 2]))
                        - compensated_image[i][j]
                    )
                    / 8
                )
    return compensated_image


def compute_compensated_frame(previous: np.ndarray, parameters: list):
    """
    Computes I' given I and the current parameters.

    Args:
        previous (np.ndarray): previous frame
        parameters (list): current parameters

    Returns:
        np.ndarray: motion-compensated frame
    """
    compensated = np.copy(previous)
    for i in range(compensated.shape[0]):
        for j in range(compensated.shape[1]):
            (x1, y1) = motion_model(parameters, i, j)
            try:
                compensated[x1][y1] = previous[i][j]
            except IndexError:
                # if out of border, we just use the old value
                pass
    return compensated


def first_estimation(precedent, current):
    """
    Computes the parameters for the perspective motion model for the first iteration.

    Since the paper does not specify how to get the parameters from the first motion estimation I assume it sets all of them to 0 but a0 and a1 which are initialized thorugh the dense motion field estimation.

    Parameters:
    @precedent  previous frame
    @current    current frame
    """
    # estimate the dense motion field
    dense_motion_field = dense_motion_estimation(precedent, current)
    parameters = compute_first_parameters(dense_motion_field)
    return parameters


def parameter_projection(parameters):
    """
    Projection of the parameters from level l to level l+1.

    Citing the paper: `The projection of the motion parameters from one level onto the next one consists merely of multiplying a0 and a1 by 2, and dividing a6 and a7 by two.`

    Args:
        parameters (list): the list of the current parameters for motion model at level l

    Returns:
        parameters (list): the list of the updated parameters for motion model at level l+1
    """
    parameters[0] *= 2
    parameters[1] *= 2
    parameters[6] /= 2
    parameters[7] /= 2
    return parameters


@timer
def handmade_gradient_descent(parameters, previous, current):
    """
    Updates the paramters with a gradient-descent-like strategy.
    start with the current estimation of the parameters
    for each parameter
        perform N iterations where
            a. compute the error compensated-real
            b. compute the delta-error
            c. change the parameter according to the delta-error
                - if delta is positive, we are going in the correct direction, keep 'er goin
                - if delta is negative, we are heading uphill, change the sign of the update
                - at each iteration decrease the value of the update

    Args:
        parameters (list):  parameters of the model of motion
        previous (np.ndarray):  previous frame
        current (np.ndarray):  current frame

    Returns:
        best (list): the optimized parameter vector for the model of motion
    """
    best = parameters
    for index in range(len(parameters)):
        step = parameters[index] / 10
        if step == 0:
            step = 0.001
        compensated = compute_compensated_frame(previous, parameters)
        error_matrix = sum_squared_differences(compensated, current)
        previous_error = np.sum(error_matrix)
        min_error = previous_error
        for _ in range(N_MAX_ITERATIONS):
            compensated = compute_compensated_frame(previous, parameters)
            error_matrix = sum_squared_differences(compensated, current)
            current_error = np.sum(error_matrix)
            if previous_error >= current_error:
                delta_ratio = current_error / previous_error
                if current_error < min_error:
                    best[index] = parameters[index]
            else:
                step = -step
                delta_ratio = previous_error / current_error
            parameters[index] += step
            step = step * delta_ratio
            previous_error = current_error
    return best


def grandient_single_parameter(parameters, previous, current, index):
    """
    Performs gradient descent on a single parameter.
    This function was created with the scope to be launched on a different process, in order to optimize the computation time of the estimation of all the parameters.

    Args:
        parameters (list):  parameters of the model of motion
        previous (np.ndarray):  previous frame
        current (np.ndarray):  current frame
        index (int): the index of the single parameter to be optimized

    Returns:
        best (list): the optimized parameter vector for the model of motion
    """
    best = parameters[index]
    step = parameters[index] / 10
    if step == 0:
        step = 0.01
    compensated = compute_compensated_frame(previous, parameters)
    error_matrix = sum_squared_differences(compensated, current)
    previous_error = np.sum(error_matrix)
    min_error = previous_error
    for _ in range(N_MAX_ITERATIONS):
        compensated = compute_compensated_frame(previous, parameters)
        error_matrix = sum_squared_differences(compensated, current)
        current_error = np.sum(error_matrix)
        if previous_error >= current_error:
            delta_ratio = current_error / previous_error
            if current_error < min_error:
                best = parameters[index]
        else:
            step = -step
            delta_ratio = previous_error / current_error
        parameters[index] += step
        step = step * delta_ratio
        previous_error = current_error
    return best


@timer
def handmade_gradient_descent_mp(parameters, previous, current):
    """
    Function used to compute the gradient descent separately for each parameter.
    The function launches a process for each parameter, each process will run a gradient descent task to optimize the parameter.

    Args:
        parameters (list):  parameters of the model of motion
        previous (np.ndarray):  previous frame
        current (np.ndarray):  current frame

    Returns:
        best (list): the optimized parameter vector for the model of motion
    """
    pool = multiprocessing.Pool(processes=len(parameters))
    args = [
        (parameters, np.copy(previous), np.copy(current), i)
        for i in range(len(parameters))
    ]
    best = pool.starmap(grandient_single_parameter, args)

    return best


def gradient_descent(parameters, previous, current):
    """
    Computing gradient descent as from Dufaux 2000.

    Args:
        parameters (list):  parameters of the model of motion
        previous (np.ndarray):  previous frame
        current (np.ndarray):  current frame

    Returns:
        best (list): the optimized parameter vector for the model of motion
    """
    compensated = compute_compensated_frame(previous, parameters)
    error_matrix = sum_squared_differences(compensated, current)
    # iterate over all the pixels, compute derivative b_k ~ e de/da_k
    a0, a1, a2, a3, a4, a5, a6, a7 = parameters
    bki = [0.0 for _ in parameters]
    b = np.zeros(shape=(len(bki),), dtype=np.float64)
    H = np.zeros(shape=(len(bki), len(bki)), dtype=np.float64)
    # double loop to compute derivatives
    for xi in range(1, previous.shape[0]):
        for yi in range(1, previous.shape[1]):
            xj, yj = motion_model(parameters, xi, yi)
            if (
                (xj < 2)
                or (xj > previous.shape[0] - 2)
                or (yj < 2)
                or (yj > previous.shape[1] - 2)
            ):
                # sanitize values for xj and yj (should be able to index +1 and -1) and should be able to divide for xj-1 and xj+1
                for k in range(len(parameters)):
                    bki[k] = 0
            else:
                # compute delta error on x direction, dex = err(xj+1) - err(xj-1)
                error_xj_plus_1 = int(previous[xj + 1][yj]) - int(current[xi][yi])
                error_xj_minus_1 = int(previous[xj - 1][yj]) - int(current[xi][yi])
                dex = error_xj_plus_1 - error_xj_minus_1
                # compute delta error on y direction, dey = err(xj+1) - err(xj-1)
                error_yj_plus_1 = int(previous[xj][yj + 1]) - int(current[xi][yi])
                error_yj_minus_1 = int(previous[xj][yj - 1]) - int(current[xi][yi])
                dey = error_yj_plus_1 - error_yj_minus_1

                # compute delta_a0 to obtain xj+1 instead of xi
                a0_plus = (a6 * xi + a7 * yi + 1) * (xj + 1) - (a2 * xi + a3 * yi)
                # compute delta_a0 to obtain xj-1 instead of xi
                a0_minus = (a6 * xi + a7 * yi + 1) * (xj - 1) - (a2 * xi + a3 * yi)
                # compute delta_a0 necessary to pass from xj-1 to xj+1
                da0 = a0_plus - a0_minus

                a1_plus = (a6 * xi + a7 * yi + 1) * (yj + 1) - (a4 * xi + a5 * yi)
                a1_minus = (a6 * xi + a7 * yi + 1) * (yj - 1) - (a4 * xi + a5 * yi)
                da1 = a1_plus - a1_minus

                a2_plus = ((a6 * xi + a7 * yi + 1) * (xj + 1) - (a3 * yi + a0)) / xi
                a2_minus = ((a6 * xi + a7 * yi + 1) * (xj - 1) - (a3 * yi + a0)) / xi
                da2 = a2_plus - a2_minus

                a3_plus = ((a6 * xi + a7 * yi + 1) * (xj + 1) - (a2 * xi + a0)) / yi
                a3_minus = ((a6 * xi + a7 * yi + 1) * (xj - 1) - (a2 * xi + a0)) / yi
                da3 = a3_plus - a3_minus

                a4_plus = ((a6 * xi + a7 * yi + 1) * (yj + 1) - (a5 * yi + a1)) / xi
                a4_minus = ((a6 * xi + a7 * yi + 1) * (yj - 1) - (a5 * yi + a1)) / xi
                da4 = a4_plus - a4_minus

                a5_plus = ((a6 * xi + a7 * yi + 1) * (yj + 1) - (a4 * xi + a1)) / yi
                a5_minus = ((a6 * xi + a7 * yi + 1) * (yj - 1) - (a4 * xi + a1)) / yi
                da5 = a5_plus - a5_minus

                a6_plus_x = ((a0 + a2 * xi + a3 * yi) / (xj + 1) - a7 * yi - 1) / xi
                a6_plus_y = ((a1 + a4 * xi + a5 * yi) / (yj + 1) - a7 * yi - 1) / xi
                a6_minus_x = ((a0 + a2 * xi + a3 * yi) / (xj - 1) - a7 * yi - 1) / xi
                a6_minus_y = ((a1 + a4 * xi + a5 * yi) / (yj - 1) - a7 * yi - 1) / xi
                da6_x = a6_plus_x - a6_minus_x
                da6_y = a6_plus_y - a6_minus_y
                da6 = da6_x
                derr6 = dex
                if da6_x < da6_y:
                    da6 = da6_y
                    derr6 = dey

                a7_plus_x = ((a0 + a2 * xi + a3 * yi) / (xj + 1) - a6 * xi - 1) / yi
                a7_plus_y = ((a1 + a4 * xi + a5 * yi) / (yj + 1) - a6 * xi - 1) / yi
                a7_minus_x = ((a0 + a2 * xi + a3 * yi) / (xj - 1) - a6 * xi - 1) / yi
                a7_minus_y = ((a1 + a4 * xi + a5 * yi) / (yj - 1) - a6 * xi - 1) / yi
                da7_x = a7_plus_x - a7_minus_x
                da7_y = a7_plus_y - a7_minus_y
                da7 = da7_x
                derr7 = dex
                if da7_x < da7_y:
                    da7 = da7_y
                    derr7 = dey

                # append derivatives to respective positions in the parameter derivatives vector
                bki[0] = dex / da0
                bki[1] = dey / da1
                bki[2] = dex / da2
                try:
                    bki[3] = dex / da3
                except:
                    bki[3] = 0
                bki[4] = dey / da4
                bki[5] = dey / da5
                bki[6] = derr6 / da6
                bki[7] = derr7 / da7
            # computing b_k and H_kl
            for k in range(len(bki)):
                # if np.isinf(bki[k]) or np.isnan(bki[k]):
                #     bki[k]=0
                b[k] += error_matrix[xi][yi] * bki[k]
                # if np.isnan(b[k]):
                #     print("AAAAAAA")
                for l in range(len(bki)):
                    H[k][l] += bki[k] * bki[l]
    b = -b
    old = np.fromiter(parameters, dtype=np.float64)
    if DIRECT_INVERSE:
        new_parameters = old + np.matmul(np.linalg.inv(H), b)
    else:
        u, s, vt = np.linalg.svd(H, full_matrices=True)
        sinv = np.copy(s)
        for i in range(len(sinv)):
            if sinv[i] != 0:
                sinv[i] = 1/sinv[i]
        S = np.identity(n=len(parameters), dtype=np.float64)
        for i in range(len(sinv)):
            S[i][i] = sinv[i]
        Hinv = np.matmul(
            np.matmul(np.transpose(vt), S), np.transpose(u)
        )
        new_parameters = old + np.matmul(Hinv, b)
    return new_parameters


@timer
def global_motion_estimation(precedent, current):
    """
    Method to perform the global motion estimation.

    Args:
        precedent (np.ndarray): the frame at time t-1
        current (np.ndarray): the frame at time t
    """
    # create the gaussian pyramids of the frames
    prev_pyr = get_pyramids(precedent)
    curr_pyr = get_pyramids(current)

    # first (coarse) level estimation
    parameters = first_estimation(prev_pyr[0], curr_pyr[0])
    parameters = gradient_descent(parameters, prev_pyr[0], curr_pyr[0])

    # all the other levels
    for i in range(1, len(prev_pyr)):
        # print("updating parameters for next level ...")
        parameters = parameter_projection(parameters)
        parameters = gradient_descent(parameters, prev_pyr[i], curr_pyr[i])
        # print(f"Updated parameters: {parameters}")
        compensated = compute_compensated_frame_complete(prev_pyr[i], parameters)
        # cv2.imshow(s, compensated)
        # cv2.waitKey(1)

    compensated = compute_compensated_frame(prev_pyr[-1], parameters)
    return compensated
