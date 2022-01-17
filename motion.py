import numpy as np
import time
from utils import get_pyramids
from bbme import Block_matcher


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
    E = E*E
    return E


def dense_motion_estimation(previous, current):
    """
        Given a couple of frames, estimates the dense motion field.

        The dense motion field corresponds to a matrix of size the # of blocks
        that fit in the image for each dimension. Each element of the matrix 
        contains two values, the shift in x and y directions.
    """
    BM = Block_matcher(block_size=6,
                       search_range=2,
                       pixel_acc=1,
                       searching_procedure=2)

    _, motion_field = BM.get_motion_field(previous, current)
    return motion_field


def compute_transition_vector(dense_motion_field):
    # need to fix this, I don't know how to compute the transition vector (as long as I don't know the form of the dense motion field)
    return (-1, -1)


def first_estimation(precedent, current):
    """
        Computes the parameters for the perspective motion model for the first iteration.
        
        Since the paper does not specify how to get the parameters from the first motion estimation I assume it sets all of them to 0 but a0 and a1 which are clearly initialized.
    """
    # estimate the dense motion field
    dense_motion_field = dense_motion_estimation(precedent, current)
    ## TODO: continque here... how do I get the first estimation?
    dx, dy = compute_transition_vector(dense_motion_field)
    a0, a1, a2, a3, a4, a5, a6, a7 = (dx, dy, 0, 0, 0, 0, 0, 0)
    return (a0, a1, a2, a3, a4, a5, a6, a7)


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


def parameter_optimization(parameters, current, previous):
    """
        Parameter optimization with gradient descent.
        The gradient is computed on the error between the motion model and the frame difference.

        Parameters:
        @parameters the vector of current parameters
        @current    the current frame
        @previous   the previous frame
    """
    pass


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
    first_estimation(prec_pyr[0], curr_pyr[0])
    
    # for i in levels_left
    ## parameters = project(parameters)
    ## until convergence or n_max iterations
    ### parameters = gradient_descent(parameters, prec_pyr, curr_pyr)
    
    # not clear what should return


if __name__ == "__main__":
    # TEST sum_squared_differences
    # cur = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # pre = np.zeros_like(cur)
    # print(type(pre))
    # E = sum_squared_differences(pre, cur)
    # print(E)

    # TEST parameter_projection
    # par = [i for i in range(0,8)]
    # print(par)
    # est = parameter_projection(par)
    # print(est)

    pass