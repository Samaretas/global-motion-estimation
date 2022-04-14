import multiprocessing
from xmlrpc.client import MAXINT
import numpy as np
from utils import timer
from utils import get_pyramids
from bbme import get_motion_field
import itertools

BBME_BLOCK_SIZE = 16
MOTION_VECTOR_ERROR_THRESHOLD_PERCENTAGE = .3


def dense_motion_estimation(previous, current):
    """Given a couple of frames, estimates the dense motion field.

    The dense motion field corresponds to a matrix of size the # of blocks
    that fit in the image for each dimension. Each element of the matrix
    contains two values, the shift in x and y directions.

    Args:
        previous (np.ndarray): previous frame of the video.
        current (np.ndarray): current frame of the video.

    Returns:
        motion_field (np.ndarray): estimated dense motion field.
    """
    motion_field = get_motion_field(
        previous, current, block_size=2, searching_procedure=3
    )
    return motion_field


def best_affine_parameters(previous, current):
    """Given two frames, computes the parameters that optimize the affine model for global motion estimation between the two frames.
    These parameters are computed minimizing a measure of error on the difference between motion vectors computed via BBME and motion vectors obtained with the affine model.
    For theoretical explanation refer to [Yao Wang, Jôrn Ostermann and Ya-Qin Zhang, Video Processing and Communications 1st Edition].

    Args:
        previous (np.ndarray): previous frame.
        current (np.ndarray): current frame.

    Returns:
        np.ndarray: array with parameters of the affine motion model [a0,a1,a2,b0,b1,b2].
    """
    # get ground truth motion field
    gt_motion_field = get_motion_field(
        previous=previous,
        current=current,
        block_size=BBME_BLOCK_SIZE,
        searching_procedure=3,
    )
    first_part = np.zeros(shape=[3, 3], dtype=np.float64)
    second_part = np.zeros(shape=[3, 1], dtype=np.float64)
    w = 1 / (previous.shape[0] * previous.shape[1])
    for i in range(gt_motion_field.shape[0]):
        for j in range(gt_motion_field.shape[1]):
            x = i * 4
            y = j * 4
            Ax = np.array([[1, x, y]], dtype=np.float64)
            dx = gt_motion_field[i, j]
            temp_first = (np.matmul(Ax.transpose(), Ax)) * w
            temp_second = (Ax.transpose() * dx[0]) * w
            first_part += temp_first
            second_part += temp_second
    finv = np.linalg.inv(np.matrix(first_part))
    finv = np.array(finv)
    ax = np.matmul(finv, second_part)

    first_part = np.zeros(shape=[3, 3], dtype=np.float64)
    second_part = np.zeros(shape=[3, 1], dtype=np.float64)
    w = 1 / (previous.shape[0] * previous.shape[1])
    for i in range(gt_motion_field.shape[0]):
        for j in range(gt_motion_field.shape[1]):
            x = i * 4
            y = j * 4
            Ay = np.array([[1, x, y]], dtype=np.float64)
            dx = gt_motion_field[i, j]
            temp_first = (np.matmul(Ay.transpose(), Ay)) * w
            temp_second = (Ay.transpose() * dx[1]) * w
            first_part += temp_first
            second_part += temp_second
    finv = np.linalg.inv(np.matrix(first_part))
    finv = np.array(finv)
    ay = np.matmul(finv, second_part)
    ax = ax.reshape((3,))
    ay = ay.reshape((3,))
    a = np.concatenate([ax, ay])
    return a


def affine_model(x, y, parameters):
    """Computes the or the displacement of the pixel in position <x,y> given by the application of the affine model with the passed parameters.

    Args:
        x (int): x coordinate of the pixel.
        y (int): y coordinate of the pixel.
        parameters (np.array): parameters of the affine model.

    Returns:
        (tuple(int, int)): displacement of the pixel in position <x,y>.
    """
    A = np.asarray([[1, x, y, 0, 0, 0], [0, 0, 0, 1, x, y]], dtype=np.int32)
    tparameters = np.transpose(parameters)
    d = np.matmul(A, tparameters)
    return d


# @timer
def global_motion_estimation(previous, current):
    """Method to perform the global motion estimation.
    - uses affine model to model global motion
    - uses robust estimation removing outliers from global motion estimation
    - uses a hierarchical approach to make the estimation more robust

    Args:
        previous (np.ndarray): the frame at time t-1.
        current (np.ndarray): the frame at time t.

    Returns:
        np.ndarray: The list of parameters of the motion model that describes the global motion between previous and current.
    """
    # create the gaussian pyramids of the frames
    prev_pyr = get_pyramids(previous)
    curr_pyr = get_pyramids(current)
    
    # first (coarse) level estimation
    parameters = np.zeros(shape=(6), dtype=np.float32)
    parameters = first_parameter_estimation(prev_pyr[0], curr_pyr[0])


    # all the other levels
    for i in range(1, len(prev_pyr)):
        parameters = parameter_projection(parameters)
        parameters = best_affine_parameters_robust(prev_pyr[i], curr_pyr[i], parameters)

    return parameters


def get_motion_field_affine(shape, parameters):
    """Computes the motion field given by the motion model.

    Args:
        shape (np.ndarray): shape of the motion field.
        parameters (np.ndarray): list of the parameters of the motion model.

    Returns:
        np.ndarray: the motion field given by the affine model with the passed parameters.
    """
    new_shape = (shape[0], shape[1], 2)
    motion_field = np.zeros(shape=new_shape, dtype=np.int16)
    for i in range(shape[0]):
        for j in range(shape[1]):
            displacement = affine_model(i, j, parameters)
            dx = round(displacement[0])
            dy = round(displacement[1])
            motion_field[i, j] = [dx, dy]
    return motion_field


def first_parameter_estimation(previous, current):
    """Computes the parameters for the perspective motion model for the first iteration.

    Parameters:
        previous:   previous frame.
        current:    current frame.

    Returns:
        np.ndarray: first estimation of the parameters, obtained with dense motion estimation.
    """
    # estimate the dense motion field
    dense_motion_field = dense_motion_estimation(previous, current)
    parameters = compute_first_parameters(dense_motion_field)
    return parameters


def compute_first_parameters(dense_motion_field):
    """Given the initial motion field, returns the first estimation of the parameters.
    For the first estimation only transition is taken into account, therefore only parameters a0 and b0 are computed.

    Args:
        dense_motion_field (np.ndarray): ndarray with shape[-1]=2; the touples in the last dimension represent the shift of the pixel from previous to current frame. It is computed via dense motion field estimation.

    Returns:
        np.ndarray: the list of parameters of the motion model.
    """
    a0 = np.mean(dense_motion_field[:, :, 0])
    b0 = np.mean(dense_motion_field[:, :, 1])
    return np.array([a0, 0.0, 0.0, b0, 0.0, 0.0], dtype=np.float32)


def parameter_projection(parameters):
    """Projection of the parameters from level l to level l+1.

    Citing the paper: `The projection of the motion parameters from one level onto the next one consists merely of multiplying a0 and a1 by 2, and dividing a6 and a7 by two.`

    Args:
        parameters (list): the list of the current parameters for motion model at level l.

    Returns:
        parameters (list): the list of the updated parameters for motion model at level l+1.
    """
    # scale transition parameters
    # a0
    parameters[0] = parameters[0] * 2
    # b0
    parameters[3] = parameters[3] * 2
    return parameters


def best_affine_parameters_robust(previous, current, old_parameters):
    """Robust version of the method to compute parameters for the affine model.
    1. estimates motion field with old parameters
    2. eliminates outliers
    3. gets new parameters without outliers

    Args:
        previous (np.ndarray): previous frame.
        current (np.ndarray): current frame.

    Returns:
        np.ndarray: array with parameters of the affine motion model [a0,a1,a2,b0,b1,b2].
    """
    # get ground truth motion field
    gt_motion_field = get_motion_field(
        previous=previous,
        current=current,
        block_size=BBME_BLOCK_SIZE,
        searching_procedure=3,
    )
    
    # get affine model's motion field w/ old parameters
    old_params_motion_field = get_motion_field_affine(gt_motion_field.shape, old_parameters)

    # compute differences and create mask to hide outliers
    # get difference between motion vectors
    diff = gt_motion_field-old_params_motion_field
    # get norm of difference vector
    diff = np.abs(diff)
    diff = diff.sum(axis=2)
    all_diffs = diff.flatten()
    all_diffs.sort()
    threshold_index = int(MOTION_VECTOR_ERROR_THRESHOLD_PERCENTAGE*len(all_diffs))
    threshold_value = all_diffs[-threshold_index] # since sort in ascending order
    outlier = diff>threshold_value

    # compute parameters minimizing error
    # use mask to avoid computing on outliers
    first_part = np.zeros(shape=[3, 3], dtype=np.float64)
    second_part = np.zeros(shape=[3, 1], dtype=np.float64)
    w = 1 / (previous.shape[0] * previous.shape[1])
    for i in range(gt_motion_field.shape[0]):
        for j in range(gt_motion_field.shape[1]):
            if not outlier[i,j]:
                x = i * 4
                y = j * 4
                Ax = np.array([[1, x, y]], dtype=np.float64)
                dx = gt_motion_field[i, j]
                temp_first = (np.matmul(Ax.transpose(), Ax)) * w
                temp_second = (Ax.transpose() * dx[0]) * w
                first_part += temp_first
                second_part += temp_second
    finv = np.linalg.inv(np.matrix(first_part))
    finv = np.array(finv)
    ax = np.matmul(finv, second_part)

    first_part = np.zeros(shape=[3, 3], dtype=np.float64)
    second_part = np.zeros(shape=[3, 1], dtype=np.float64)
    w = 1 / (previous.shape[0] * previous.shape[1])
    for i in range(gt_motion_field.shape[0]):
        for j in range(gt_motion_field.shape[1]):
            if not outlier[i,j]:
                x = i * 4
                y = j * 4
                Ay = np.array([[1, x, y]], dtype=np.float64)
                dx = gt_motion_field[i, j]
                temp_first = (np.matmul(Ay.transpose(), Ay)) * w
                temp_second = (Ay.transpose() * dx[1]) * w
                first_part += temp_first
                second_part += temp_second
    finv = np.linalg.inv(np.matrix(first_part))
    finv = np.array(finv)
    ay = np.matmul(finv, second_part)
    ax = ax.reshape((3,))
    ay = ay.reshape((3,))
    a = np.concatenate([ax, ay])
    return a


def compensate_frame(frame, motion_field):
    """Compute the compensated frame given the starting frame and the motion field. Assumes the motion field was computed on squared blocks.

    Args:
        frame (np.ndarray): frame to be compensated.
        motion_field (np.ndarray): motion field.
    
    Returns:
        np.ndarray: the compensated frame.
    """
    X = 1
    Y = 0    
    compensated = np.copy(frame)
    # compute size of blocks
    bs = frame.shape[0]//motion_field.shape[0]
    for i in range(motion_field.shape[0]):
        for j in range(motion_field.shape[1]):
            a = i*bs
            d = motion_field[i,j]
            for _ in range(bs):
                b = j*bs
                for __ in range(bs):
                    try:
                        newa = a-d[X]
                        newb = b-d[Y]
                        assert newa > -1
                        assert newb > -1
                        compensated[a,b] = frame[newa,newb]
                    except:
                        pass
                    b += 1
                a += 1
    return compensated


def motion_compensation(previous, current):
    """Computes the compensated frame given the previous and the current frames.

    Args:
        previous (np.ndarray): previous frame.
        current (np.ndarray): current frame.

    Returns:
        np.ndarray: previous frame compensated w.r.t. the global motion estimated within the two frames.
    """
    # parameters of the affine model
    parameters = global_motion_estimation(previous, current)
    shape = (previous.shape[0]//BBME_BLOCK_SIZE, previous.shape[1]//BBME_BLOCK_SIZE)
    # motion field due to global motion
    motion_field = get_motion_field_affine(shape, parameters)
    # compensate camera motion on previous frame
    compensated = compensate_frame(previous, motion_field)
    return compensated