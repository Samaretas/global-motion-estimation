from math import floor
import os
import cv2
import itertools
import numpy as np
from utils import draw_motion_vector, get_video_frames


def get_motion_fied(previous, current, block_size=4, search_window=2, searching_procedure=1):
    height = previous.shape[0]
    width = previous.shape[1]

    motion_field = np.empty(
        (int(height / block_size), int(width / block_size), 2))

    if searching_procedure == 1:
        motion_field = exhaustive_search(previous, current, motion_field, height,
                                         width,  mae, block_size, search_window)
    if searching_procedure == 2:
        motion_field = threestep_search(previous, current, motion_field, height,
                                        width,  mae, block_size, search_window)
    return motion_field


def compute_dfd(current_block, anchor_block, pnorm_function):
    """
    Computes a given pnorm distance of two np arrays

    Parameters:
    @current_block 
    @anchor_block
    @pnorm_function name of the function to use
    """
    assert current_block.shape == anchor_block.shape
    diff_block = np.array(current_block, dtype=np.float16) - np.array(anchor_block, dtype=np.float16)
    return pnorm_function(diff_block)


def mae(diff_block):
    """
    Given a np array, returns the sum of absolute value
    (for 1-norm distance)

    Parameters:
    @diff_block block to compute
    """
    return np.sum(np.abs(diff_block))


def mse(diff_block):
    """
    Given a np array, returns the sum of squares
    (for 2-norm distance)

    Parameters:
    @diff_block block to compute
    """
    return np.sum(diff_block * diff_block)


def exhaustive_search(previous, current, mf, height, width, pnorm_function, block_size=4, search_window=2):
    """
    Exhaustive search

    Parameters:
    @previous
    @current
    @mf np array of the motion field to compute
    @height rows of the frame
    @width columns of the frame
    @pnorm_function dfd function to use
    @block_size size of each block (in pixels)
    @search_windows halvened dimension of the search window
    """
    # print(block_size, search_window, height, width)
    # iterates for all blocks of the current frames
    for (block_row, block_col) in itertools.product(range(0, height - (block_size - 1), block_size),
                                                    range(0, width - (block_size - 1), block_size)):

        anchor_block = previous[block_row: block_row + block_size,
                                block_col: block_col + block_size]

        # initialize minimum
        min_block = np.infty

        # scan the search window
        for (window_col, window_row) in itertools.product(range(-search_window, (search_window + block_size)),
                                                          range(-search_window, (search_window + block_size))):
            # check if block centered at the current pixel in the search window
            # falls entirely  within the image grid
            top_left_y = block_row + window_row
            top_left_x = block_col + window_col
            bottom_right_y = block_row + window_row + block_size - 1
            bottom_right_x = block_col + window_col + block_size - 1

            if not (top_left_y < 0 or
                    top_left_x < 0 or
                    bottom_right_y > height - 1 or
                    bottom_right_x > width - 1):
                # get the block centered at the current pixel in the search window from the4 current frame
                current_block = current[top_left_y: bottom_right_y +
                                        1, top_left_x: bottom_right_x + 1]
                dfd = compute_dfd(current_block, anchor_block, pnorm_function)

                # if a new minimum is found, update minimum and save coordinates
                if dfd < min_block:
                    min_block = dfd
                    min_x = window_row
                    min_y = window_col

        mf[floor(block_row / block_size),
           floor(block_col / block_size), 0] = min_y
        mf[floor(block_row / block_size),
           floor(block_col / block_size), 1] = min_x

    print('exhaustive done')
    return mf


def threestep_search(previous, current, mf, height, width, pnorm_function, block_size=4, search_window=2):
    """
    Three-step search

    Parameters:
    @previous
    @current
    @mf np array of the motion field to compute
    @height rows of the frame
    @width columns of the frame
    @pnorm_function dfd function to use
    @block_size size of each block (in pixels)
    @search_windows halvened dimension of the search window
    """

    step_one = int(((2 * search_window) + block_size) / 3)
    step_two = int(((2 * search_window) + block_size) / 5)
    step_three = int(((2 * search_window) + block_size) / 10)
    dx, dy = 0, 0
    for (block_row, block_col) in itertools.product(range(0, height - (block_size - 1), block_size),
                                                    range(0, width - (block_size - 1), block_size)):

        anchor_block = previous[block_row: block_row + block_size,
                                block_col: block_col + block_size]

        # initialize minimum
        min_block = np.infty

        # Executing first step
        for (window_col, window_row) in itertools.product([-step_one, 0, step_one], [-step_one, 0, step_one]):
            # Compute current target block vertices
            top_left_y = block_row + window_row
            top_left_x = block_col + window_col
            bottom_right_y = block_row + window_row + block_size - 1
            bottom_right_x = block_col + window_col + block_size - 1

            if not (top_left_y < 0 or
                    top_left_x < 0 or
                    bottom_right_y > height - 1 or
                    bottom_right_x > width - 1):

                # get the block centered at the current pixel in the search window from the4 current frame
                current_block = current[top_left_y: bottom_right_y + 1,
                                        top_left_x: bottom_right_x + 1]

                dfd = compute_dfd(current_block, anchor_block, pnorm_function)

                # if a new minimum is found, update minimum and save coordinates
                if dfd < min_block:
                    min_block = dfd
                    dx = window_row
                    dy = window_col

        # Compute new origin
        block_row_step_2 = block_row + dx
        block_col_step_2 = block_col + dy

        # Executing second step
        for (window_col, window_row) in itertools.product([-step_two, 0, step_two], [-step_two, 0, step_two]):
            # Compute current target block vertices
            top_left_y = block_row_step_2 + window_row
            top_left_x = block_col_step_2 + window_col
            bottom_right_y = block_row_step_2 + window_row + block_size - 1
            bottom_right_x = block_col_step_2 + window_col + block_size - 1

            if not (top_left_y < 0 or
                    top_left_x < 0 or
                    bottom_right_y > height - 1 or
                    bottom_right_x > width - 1):

                # get the block centered at the current pixel in the search window from the4 current frame
                current_block = current[top_left_y: bottom_right_y +
                                        1, top_left_x: bottom_right_x + 1]
                dfd = compute_dfd(current_block, anchor_block, pnorm_function)

                print(dfd)
                # if a new minimum is found, update minimum and save coordinates
                if dfd < min_block:
                    min_block = dfd
                    dx = window_row
                    dy = window_col

        # Compute new origin
        block_row_step_3 = block_row_step_2 + dx
        block_col_step_3 = block_col_step_2 + dy

        # Executing third step
        for (window_col, window_row) in itertools.product([-step_three, 0, step_three], [-step_three, 0, step_three]):
            # Compute current target block vertices
            top_left_y = block_row_step_3 + window_row
            top_left_x = block_col_step_3 + window_col
            bottom_right_y = block_row_step_3 + window_row + block_size - 1
            bottom_right_x = block_col_step_3 + window_col + block_size - 1

            if not (top_left_y < 0 or
                    top_left_x < 0 or
                    bottom_right_y > height - 1 or
                    bottom_right_x > width - 1):

                # get the block centered at the current pixel in the search window from the4 current frame
                current_block = current[top_left_y: bottom_right_y +
                                        1, top_left_x: bottom_right_x + 1]
                dfd = compute_dfd(current_block, anchor_block, pnorm_function)

                # if a new minimum is found, update minimum and save coordinates
                if dfd < min_block:
                    min_block = dfd
                    dx = window_row
                    dy = window_col

        mf[floor(block_row / block_size),
           floor(block_col / block_size), 0] = dy
        mf[floor(block_row / block_size),
           floor(block_col / block_size), 1] = dx

    return mf

    # print('3steps')


def twodlog_search():
    print('2dlog')


if __name__ == '__main__':
    frames = get_video_frames("./hall_objects_qcif.y4m")
    idx = 30

    previous = frames[idx-1]
    current = frames[idx]

    # print(previous.shape)
    # cv2.imshow("cur", current)
    # cv2.waitKey(0)

    motion_field = get_motion_fied(
        previous, current, block_size=6, searching_procedure=2)
    # print(motion_field)
    draw = draw_motion_vector(current, motion_field)

    cv2.imwrite(os.path.join('mv_drawing.png'), draw)
    # print("motion_field")
    # print(type(motion_field))
    # print(motion_field.shape)
    # print(motion_field.max)
    # print(motion_field)
