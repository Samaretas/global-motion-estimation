import os
import cv2
import itertools
import numpy as np
from math import floor
from scipy.ndimage.filters import gaussian_filter
from utils import draw_motion_vector, get_video_frames


def get_motion_fied(previous, current, block_size=4, search_window=2, searching_procedure=1, pnorm_distance=0):
    height = previous.shape[0]
    width = previous.shape[1]

    motion_field = np.empty(
        (int(height / block_size), int(width / block_size), 2))

    # if searching_procedure == 1:
    #     motion_field = exhaustive_search(previous, current, motion_field, height,
    #                                      width,  mae, block_size, search_window)
    # if searching_procedure == 2:
    #     motion_field = threestep_search(previous, current, motion_field, height,
    #                                     width,  mae, block_size, search_window)
    print(motion_field.shape)
    search = searching_procedures[searching_procedure]
    pnorm = pnorm_distances[pnorm_distance]
    motion_field = search(previous, current, motion_field, height,
                          width,  pnorm, block_size, search_window)
    return motion_field


def compute_dfd(current_block, anchor_block, pnorm_function):
    """
    compute_dfd 
    Computes a given pnorm distance of two np arrays

    Args:
        current_block (np.ndarray): [description]
        anchor_block (np.ndarray): [description]
        pnorm_function (func): name of the function to use

    Returns:
        float: value of the dfd
    """

    assert current_block.shape == anchor_block.shape
    diff_block = np.array(current_block, dtype=np.float16) - \
        np.array(anchor_block, dtype=np.float16)
    return pnorm_function(diff_block)

def mae(diff_block):
    """
    mae 
    Given a np array, returns the sum of absolute value
    (for 1-norm distance)

    Args:
        diff_block (np.ndarray): the block to sum

    Returns:
        float: value of the sum of absolute values
    """
    return np.sum(np.abs(diff_block))


def mse(diff_block):
    """
    mse 
    Given a np array, returns the sum of eucledian distance
    (for 2-norm distance)

    Args:
        diff_block (np.ndarray): the block to sum

    Returns:
        float: value of the sum of eucledian distance
    """
    return np.sum(diff_block * diff_block)


def compute_current_target_block_corners(br, bl, wr, wc, bs):
    top_left_x = wr
    top_left_y = wc
    bottom_right_x = wr + bs - 1
    bottom_right_y = wc + bs - 1
    return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)


def exhaustive_search(previous, current, mf, height, width, pnorm_function, block_size=4, search_window=2):
    """
    exhaustive_search 

    Args:
        previous (np.ndarray): previous frame
        current (np.ndarray): current frame
        mf (np.ndarray): motion field to compute
        height (int): rows of the frame
        width (int): columns of the frame
        pnorm_function (func): dfd function to use
        block_size (int, optional): size of each block (in pixels). Defaults to 4.
        search_window (int, optional): sized of the search window (in pixels). Defaults to 2.

    Returns:
        np.ndarray: motion field
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
    threestep_search 

    Args:
        previous (np.ndarray): previous frame
        current (np.ndarray): current frame
        mf (np.ndarray): motion field to compute
        height (int): rows of the frame
        width (int): columns of the frame
        pnorm_function (func): dfd function to use
        block_size (int, optional): size of each block (in pixels). Defaults to 4.
        search_window (int, optional): sized of the search window (in pixels). Defaults to 2.

    Returns:
        np.ndarray: motion field
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

            print(f"current position {window_row}, {window_col}")
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
           floor(block_col / block_size), 1] = dy
        mf[floor(block_row / block_size),
           floor(block_col / block_size), 0] = dx

    return mf

    # print('3steps')


def twodlog_search(previous, current, mf, height, width, pnorm_function, block_size=4, search_window=2):
    positions = []
    print(height, width)
    out_of_bound = False
    for (block_row, block_col) in itertools.product(range(0, height - (block_size - 1), block_size),
                                                    range(0, width - (block_size - 1), block_size)):
        dx, dy = 0, 0
        step_size = search_window
        print(f"block {block_row} {block_col}")

        anchor_block = previous[block_row: block_row + block_size,
                                block_col: block_col + block_size]


        # initialize block minimum
        min_block = np.infty

        # get coordinates of current block
        x, y = block_row, block_col

        while step_size > 1:
            print(f"\tcurrent origin {x} {y}")

            # initialize step minimum
            min_step = min_block

            positions.clear()
            if step_size > 2:
                # search positions are cross-shaped
                positions.append([x, y])
                positions.append([x + step_size * block_size, y])
                positions.append([x - step_size * block_size, y])
                positions.append([x, y + step_size * block_size])
                positions.append([x, y - step_size * block_size])
            elif step_size == 2:
                positions = list(itertools.product([x - 2 * block_size, x, x + 2 * block_size],
                                                   [y - 2 * block_size, y, y + 2 * block_size]))

            for (window_row, window_col) in positions:
                print(f"\t\tcurrent position {window_row}, {window_col}")
                # print(f"\t\tx and y: {x} {y}")

                top_left, bottom_right = compute_current_target_block_corners(
                    x, y, window_row, window_col, block_size)

                if (top_left[0] < 0 or top_left[1] < 0 or
                        bottom_right[0] > width - 1 or bottom_right[1] > height - 1):
                    print(f"\t\t\ttop left: {top_left}, bottom right: {bottom_right}")
                    print("\t\t\t!! out of bound !!")
                    out_of_bound = True
                    continue

                current_block = current[top_left[1]: bottom_right[1] +
                                        1, top_left[0]: bottom_right[0] + 1]
                dfd = compute_dfd(current_block, anchor_block, pnorm_function)
                print(f"\t\t{dfd}")

                if dfd < min_step:
                    print("\t\tupdate min")
                    min_step = dfd
                    min_block = dfd
                    dx = window_row
                    dy = window_col

            # if out_of_bound:
            #     out_of_bound = False
            #     dx, dy = x, y
            #     break

            if dx == x and dy == y or step_size == 2:
                print("\tcount")
                # update step size
                step_size //= 2
            print("")
            

            # update coordinates
            x, y = dx, dy
            print(step_size)
            print(min_block)
            print(f"min positions: {dx}, {dy}")
            print("")
            print("")

        print(f"saved positions: {dx}, {dy}")
        print("")
        print("-------------")
        print("")
        mf[block_row // block_size,
            block_col // block_size, 0] = dx - block_row
        mf[block_row // block_size,
            block_col // block_size, 1] = dy - block_col

    return mf


def diamond_search(previous, current, mf, height, width, pnorm_function, block_size, masked = False):

    motion_map = np.zeros((height, width, 2), dtype='int16')
    previous_motion = None

    # notice here the approach at borders, at the moment we neglect right and bottom leftovers
    for (row, col) in itertools.product(range(0, height - block_size, block_size),
                                        range(0, width - block_size, block_size)):
    # for row in range(0, height - block_size, block_size):
    #     for col in range(0, width-block_size, block_size):
            # iterating over all image positions
        shift_distance = -1
        
        anchor_block = previous[row:row + block_size, col:col + block_size]
        match_position = (row, col)
        path = [match_position]

        large_search_pattern_offsets = [
            (0, 0),
            (2, 0),
            (1, 1),
            (0, 2),
            (-1, 1),
            (-2, 0),
            (-1, -1),
            (0, -2),
            (1, -1)
        ]
        small_search_pattern_offsets = [
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1)
        ]

        # large diamond search
        stopping_condition = False
        while not stopping_condition:
            shift_distance += 1
            min_diff = np.infty
            best_pos = match_position
            for offset in large_search_pattern_offsets:
                (row2, col2) = (match_position[0] + offset[0],
                                match_position[1] + offset[1])
                row2 = min(max(row2, 0), height - block_size - 1)
                col2 = min(max(col2, 0), width - block_size - 1)
                block = current[row2:row2 + block_size,
                                col2:col2 + block_size]
                diff = compute_dfd(anchor_block, block, pnorm_function)

                if diff < min_diff:
                    min_diff = diff
                    best_pos = (row2, col2)

            stopping_condition = (match_position == best_pos)
            path.append(best_pos)
            match_position = (best_pos)

        # small diamond search
        min_diff = np.infty
        for offset in small_search_pattern_offsets:
            (row2, col2) = (match_position[0] + offset[1],
                            match_position[1] + offset[0])
            row2 = min(max(row2, 0), height - block_size - 1)
            col2 = min(max(col2, 0), width - block_size - 1)
            block = current[row2:row2 + block_size,
                            col2:col2 + block_size]
            diff = compute_dfd(anchor_block, block, pnorm_function)

            if diff < min_diff:
                min_diff = diff
                best_pos = (row2, col2)

        if best_pos != match_position:
            shift_distance += 1
            path.append(best_pos)

        # Add motion to all the positions matched, motion is proportional to the amount of shift
        if shift_distance:
            motion_map[row, col, 0] = best_pos[0]
            motion_map[row, col, 1] = best_pos[1]
            print(row, col)
            print("")
            # for position in path:
            #     motion_map[position[0], position[1], 0] += shift_distance

    # Blurring
    # gauss_rounds = 3
    # for _ in range(gauss_rounds):
    #     motion_map = cv2.GaussianBlur(
    #         motion_map, (block_size, block_size), ((block_size - 1) / 6))
    
    # Normalization
    map_max = np.max(motion_map)
    normalized = motion_map.astype('float32') / map_max

    # Thresholding
    threshold = 1 / 3
    def thresher(x): return (x * (255 / map_max)) if x > threshold else 0
    vthresher = np.vectorize(thresher)
    normalized = vthresher(normalized)
    if(masked):
        if previous_motion is None:
            previous_motion = (normalized.astype(bool)).astype('uint8')
        else:
            temp = normalized.astype(bool)
            normalized *= previous_motion
            previous_motion = temp

    return normalized.astype('float32')


pnorm_distances = [mae, mse]
searching_procedures = [exhaustive_search, threestep_search, twodlog_search, diamond_search]

if __name__ == '__main__':
    frames = get_video_frames("./hall_objects_qcif.y4m")
    idx = 30

    previous = frames[idx-20]
    current = frames[idx]

    # print(previous.shape)
    # cv2.imshow("cur", current)
    # cv2.waitKey(0)

    motion_field = get_motion_fied(
        previous, current, block_size=6, searching_procedure=3, search_window=2)
    print(motion_field)
    
    # mf = np.zeros(
    #     (int(previous.shape[0] / 6), int(previous.shape[1] / 6), 2))
    # mf[0//6,18//6,0] = 12  - 0
    # mf[0//6,18//6,1] =  6 - 18
    # draw = draw_motion_vector(current, motion_field)
    # cv2.imwrite(os.path.join('mv_drawing.png'), draw)
    # draw = draw_motion_vector(current, mf)
    # cv2.imwrite(os.path.join('fake_drawing.png'), draw)

    # print("motion_field")
    # print(type(motion_field))
    # print(motion_field.shape)
    # print(motion_field.max)
    # print(motion_field)
