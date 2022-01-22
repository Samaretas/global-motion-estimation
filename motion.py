import multiprocessing
from xmlrpc.client import MAXINT
import numpy as np
import math
from utils import get_pyramids
from bbme import Block_matcher

N_MAX_ITERATIONS = 100


def sum_squared_differences(pre_frame, cur_frame):
    """
    Computes the sum of squared differences between frames.

    Parameters:
    @pre_frame  previous frame
    @cur_frame  current frame
    """
    if type(pre_frame) != np.ndarray or type(cur_frame) != np.ndarray:
        raise Exception("Should use only numpy arrays as parameters")
    if pre_frame.shape != cur_frame.shape:
        raise Exception("Frames must have the same shape")
    E = cur_frame - pre_frame
    E = E * E
    return E


def dense_motion_estimation(previous, current):
    """
    Given a couple of frames, estimates the dense motion field.

    The dense motion field corresponds to a matrix of size the # of blocks
    that fit in the image for each dimension. Each element of the matrix
    contains two values, the shift in x and y directions.
    """
    BM = Block_matcher(block_size=6, search_range=2, pixel_acc=1, searching_procedure=2)

    _, motion_field = BM.get_motion_field(previous, current)
    return motion_field


def compute_first_parameters(dense_motion_field):
    """
    Given the initial motion field, returns the first estimation of the parameters.
    """
    a0 = np.mean(dense_motion_field[:, :, 0])
    a1 = np.mean(dense_motion_field[:, :, 1])
    #! first initialization needs a2 and a5 to be 1, otherwise the prediction will be completely blank
    (a2, a3, a4, a5, a6, a7) = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    return [a0, a1, a2, a3, a4, a5, a6, a7]


def motion_model(p, x, y):
    """
    Given the current parameters and the coordinates of a point, computes the compensated coordinates.
    """
    # TODO: change this with a more conscious approximation (<> x.5)
    try:
        x1 = int((p[0] + p[2] * x + p[3] * y) / (p[6] * x + p[7] * y + 1))
        y1 = int((p[1] + p[4] * x + p[5] * y) / (p[6] * x + p[7] * y + 1))
    except:
        # print(f"Denominator problem: {(p[6]*x+p[7]*y+1)} cannot be used")
        # print(f"parameters p[6]:{p[6]} p[7]:{p[7]} x:{x} y:{y}")
        x1 = y1 = MAXINT
    return (x1, y1)


def compute_compensated_frame(pre_frame, parameters):
    """
    Computes I' given I and the current parameters.
    """
    compensated = np.zeros_like(pre_frame)
    for i in range(compensated.shape[0]):
        for j in range(compensated.shape[1]):
            (x1, y1) = motion_model(parameters, i, j)
            # sanitize limits
            x1 = max(0, x1)
            x1 = min(x1, compensated.shape[0] - 1)
            y1 = max(0, y1)
            y1 = min(y1, compensated.shape[1] - 1)
            compensated[x1][y1] = pre_frame[i][j]
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

    `The projection of the motion parameters from one level onto the next one consists merely of multiplying a0 and a1 by 2, and dividing a6 and a7 by two. `
    """
    parameters[0] *= 2
    parameters[1] *= 2
    parameters[6] /= 2
    parameters[7] /= 2
    return parameters


def handmade_gradient_descent(parameters, previous, current):
    """
    0. start with the current estimation of the parameters
    for each parameter
        perform N iterations where
            a. compute the error compensated-real
            b. compute the delta-error
            c. change the parameter according to the delta-error
                - if delta is positive, we are going in the correct direction, keep 'er goin
                - if delta is negative, we are heading uphill, change the sign of the update
                - at each iteration decrease the value of the update
    Note: here we can apply massively parallelism 
    """
    # fh = open("./log", "w")
    best = parameters
    for p in range(len(parameters)):
        # fh.write(f"\nparameter a{p}\n")
        step = parameters[p]/10
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
                delta_ratio = current_error/previous_error
                if current_error < min_error:
                    best = parameters
            else:
                step = -step
                delta_ratio = previous_error/current_error
            parameters[p] += step
            step = step*delta_ratio
            # fh.write(f"error change from: {previous_error} to: {current_error}, updated step: {step}, current a{p} value: {parameters[p]}\n")
            previous_error=current_error
    # fh.close()
    return best


def grandient_single_parameter(parameters, previous, current, index):
    best = parameters[index]
    step = parameters[index]/10
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
            delta_ratio = current_error/previous_error
            if current_error < min_error:
                best = parameters[index]
        else:
            step = -step
            delta_ratio = previous_error/current_error
        parameters[index] += step
        step = step*delta_ratio
        previous_error=current_error
    return best

def handmade_gradient_descent_mp(parameters, previous, current):
    """
    One process for each parameter.
    """
    # best = parameters
    # processes = list()
    # for p in range(len(parameters)):
    #     process = Process(target=grandient_single_parameter, args=(parameters, previous, current, p))
    #     processes.append(process)
    
    # for p in range(len(parameters)):
    #     processes[p].start()

    # for p in range(len(parameters)):
    #     best[p] = processes[p].join()
    pool = multiprocessing.Pool(processes=len(parameters))
    args = [(parameters,previous, current, i) for i in range(len(parameters))]
    best = pool.starmap(grandient_single_parameter, args)

    return best



def global_motion_estimation(precedent, current):
    """
    Method to perform the global motion estimation.

    Parameters:
        @precedent  the frame at time t-1
        @current    the frame at time t
    """
    # create the gaussian pyramids of the frames
    prec_pyr = get_pyramids(precedent)
    curr_pyr = get_pyramids(current)

    # first (coarse) level estimation
    parameters = first_estimation(prec_pyr[0], curr_pyr[0])
    parameters = handmade_gradient_descent(parameters, prec_pyr[0], curr_pyr[0])

    # all the other levels
    for i in range(1, len(prec_pyr)):
        parameters = parameter_projection(parameters)
        parameters = handmade_gradient_descent_mp(parameters, prec_pyr[i], curr_pyr[i])
    
    compensated = compute_compensated_frame(prec_pyr[-1], parameters)
    actual_motion = np.absolute(compensated.astype('int')-current.astype('int'))
    actual_motion = actual_motion.astype('uint8')
    return actual_motion
