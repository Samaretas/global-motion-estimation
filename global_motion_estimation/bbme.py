import os
import argparse
import itertools
from xmlrpc.client import MAXINT

import cv2
import numpy as np

from utils import draw_motion_field, get_video_frames, get_pyramids


def get_motion_field(
    previous,
    current,
    block_size=4,
    search_window=2,
    searching_procedure=1,
    pnorm_distance=1,
) -> np.ndarray:
    height = previous.shape[0]
    width = previous.shape[1]

    motion_field = np.zeros(
        (int(height / block_size), int(width / block_size), 2), dtype=np.int32
    )

    search = searching_procedures[searching_procedure]
    motion_field = search(
        previous,
        current,
        motion_field,
        height,
        width,
        pnorm_distance,
        block_size,
        search_window,
    )
    return motion_field


def compute_dfd(block_1, block_2, pnorm_index=0):
    """
    compute_dfd
    Computes a given pnorm distance of two np arrays

    Args:
        block_1 (np.ndarray): first block
        block_2 (np.ndarray): the other block
        pnorm_index (func): index of the distance to use:
            0: mae
            1: mse
            Defaults to 0 (mae)


    Returns:
        float: value of the dfd
    """

    assert block_1.shape == block_2.shape
    pnorm = pnorm_distances[pnorm_index]
    diff_block = np.array(block_1, dtype=np.float32) - np.array(
        block_2, dtype=np.float32
    )
    return pnorm(diff_block)


def mae(diff_block):
    """
    mae
    Given a np array, returns the sum of of the
    minimum absolute error (1-norm distance)

    Args:
        diff_block (np.ndarray): the block to sum

    Returns:
        float: value of the sum of absolute values
    """
    return np.sum(np.abs(diff_block))


def mse(diff_block):
    """
    mse
    Given a np array, returns the sum of of the
    minimum squared  error (2-norm distance)

    Args:
        diff_block (np.ndarray): the block to sum

    Returns:
        float: value of the sum of eucledian distance
    """
    return np.sum(diff_block * diff_block)


def compute_current_target_block_corners(br, bl, wr, wc, bs):
    top_left_y = wr
    top_left_x = wc
    bottom_right_y = wr + bs - 1
    bottom_right_x = wc + bs - 1
    return (top_left_y, top_left_x), (bottom_right_y, bottom_right_x)


def exhaustive_search(
    previous,
    current,
    mf,
    height,
    width,
    pnorm_distance=0,
    block_size=4,
    search_window=2,
):
    """
    exhaustive_search
    Computes the motion field with the Exhaustive BBME algorithm

    Args:
        previous (np.ndarray): previous frame
        current (np.ndarray): current frame
        mf (np.ndarray): motion field to return
        height (int): rows of the frame
        width (int): columns of the frame
        pnorm_distance (int, optional): index of the dfd function to use. Defaults to 0 (mae)
        block_size (int, optional): size of each block (in pixels). Defaults to 4.
        search_window (int, optional): sized of the search window (in pixels). Defaults to 2.

    Returns:
        np.ndarray: motion field
    """
    # iterates for all blocks of the current frames
    for (block_row, block_col) in itertools.product(
        range(0, height - (block_size - 1), block_size),
        range(0, width - (block_size - 1), block_size),
    ):

        anchor_block = previous[
            block_row : block_row + block_size, block_col : block_col + block_size
        ]

        # initialize minimum
        min_block = np.infty

        # scan the search window
        for (window_col, window_row) in itertools.product(
            range(-search_window, (search_window + block_size)),
            range(-search_window, (search_window + block_size)),
        ):
            # check if block centered at the current pixel in the search window
            # falls entirely  within the image grid
            top_left_y = block_row + window_row
            top_left_x = block_col + window_col
            bottom_right_y = top_left_y + block_size - 1
            bottom_right_x = top_left_x + block_size - 1

            if not (
                top_left_y < 0
                or top_left_x < 0
                or bottom_right_y > height - 1
                or bottom_right_x > width - 1
            ):

                # get the block centered at the current pixel in the search window from the current frame
                current_block = current[
                    top_left_y : bottom_right_y + 1, top_left_x : bottom_right_x + 1
                ]
                dfd = compute_dfd(current_block, anchor_block, pnorm_distance)

                # if a new minimum is found, update minimum and save coordinates
                if dfd < min_block:
                    min_block = dfd
                    dx = window_row
                    dy = window_col

        mf[block_row // block_size, block_col // block_size, 0] = dy
        mf[block_row // block_size, block_col // block_size, 1] = dx

    return mf


def threestep_search(
    previous,
    current,
    mf,
    height,
    width,
    pnorm_distance=0,
    block_size=4,
    search_window=12,
):
    """
    threestep_search
    Computes the motion field with the three step search BBME algorithm

    Args:
        previous (np.ndarray): previous frame
        current (np.ndarray): current frame
        mf (np.ndarray): motion field to compute
        height (int): rows of the frame
        width (int): columns of the frame
        pnorm_distance (int, optional): index of the dfd function to use. Defaults to 0 (mae)
        block_size (int, optional): size of each block (in pixels). Defaults to 4.
        search_window (int, optional): sized of the search window (in pixels). Defaults to 12.

    Returns:
        np.ndarray: motion field
    """

    # compute step size according to the search window
    step_one = int(((2 * search_window) + block_size) / 3)
    step_two = int(((2 * search_window) + block_size) / 5)
    step_three = int(((2 * search_window) + block_size) / 10)

    dx, dy, tmp_dx, tmp_dy = 0, 0, 0, 0
    for (block_row, block_col) in itertools.product(
        range(0, height - (block_size - 1), block_size),
        range(0, width - (block_size - 1), block_size),
    ):

        anchor_block = previous[
            block_row : block_row + block_size, block_col : block_col + block_size
        ]

        # initialize minimum
        min_block = np.infty

        # executing first step
        for (window_col, window_row) in itertools.product(
            [-step_one, 0, step_one], [-step_one, 0, step_one]
        ):

            # Compute current target block vertices
            top_left_y = block_row + window_row
            top_left_x = block_col + window_col
            bottom_right_y = top_left_y + block_size - 1
            bottom_right_x = top_left_x + block_size - 1

            if not (
                top_left_y < 0
                or top_left_x < 0
                or bottom_right_y > height - 1
                or bottom_right_x > width - 1
            ):

                # get the block centered at the current pixel in the search window from the current frame
                current_block = current[
                    top_left_y : bottom_right_y + 1, top_left_x : bottom_right_x + 1
                ]

                dfd = compute_dfd(current_block, anchor_block, pnorm_distance)

                # if a new minimum is found, update minimum and save coordinates
                if dfd < min_block:
                    min_block = dfd
                    dx = window_row
                    dy = window_col

        # Compute new origin
        block_row_step_2 = block_row + dx
        block_col_step_2 = block_col + dy

        # reinitialize minimum
        min_block = np.infty

        # Executing second step
        for (window_col, window_row) in itertools.product(
            [-step_two, 0, step_two], [-step_two, 0, step_two]
        ):
            # Compute current target block vertices
            top_left_y = block_row_step_2 + window_row
            top_left_x = block_col_step_2 + window_col
            bottom_right_y = top_left_y + block_size - 1
            bottom_right_x = top_left_x + block_size - 1

            if not (
                top_left_y < 0
                or top_left_x < 0
                or bottom_right_y > height - 1
                or bottom_right_x > width - 1
            ):

                # get the block centered at the current pixel in the search window from the4 current frame
                current_block = current[
                    top_left_y : bottom_right_y + 1, top_left_x : bottom_right_x + 1
                ]
                dfd = compute_dfd(current_block, anchor_block, pnorm_distance)

                # print(dfd)
                # if a new minimum is found, update minimum and save coordinates
                if dfd < min_block:
                    min_block = dfd
                    tmp_dx = window_row
                    tmp_dy = window_col

        dx += tmp_dx
        dy += tmp_dy

        # compute new origin
        block_row_step_3 = block_row_step_2 + dx
        block_col_step_3 = block_col_step_2 + dy

        # reinitialize minimum
        min_block = np.infty

        # executing third step
        for (window_col, window_row) in itertools.product(
            [-step_three, 0, step_three], [-step_three, 0, step_three]
        ):
            # compute current target block vertices
            top_left_y = block_row_step_3 + window_row
            top_left_x = block_col_step_3 + window_col
            bottom_right_y = block_row_step_3 + window_row + block_size - 1
            bottom_right_x = block_col_step_3 + window_col + block_size - 1

            if not (
                top_left_y < 0
                or top_left_x < 0
                or bottom_right_y > height - 1
                or bottom_right_x > width - 1
            ):

                # get the block centered at the current pixel in the search window from the4 current frame
                current_block = current[
                    top_left_y : bottom_right_y + 1, top_left_x : bottom_right_x + 1
                ]
                dfd = compute_dfd(current_block, anchor_block, pnorm_distance)

                # if a new minimum is found, update minimum and save coordinates
                if dfd < min_block:
                    min_block = dfd
                    tmp_dx = window_row
                    tmp_dy = window_col

        dx += tmp_dx
        dy += tmp_dy

        mf[block_row // block_size, block_col // block_size, 0] = dy
        mf[block_row // block_size, block_col // block_size, 1] = dx

    return mf


def twodlog_search(
    previous, current, mf, height, width, pnorm_function, block_size=4, search_window=12
):
    """
    twodlog_search
    Computes the motion field with the 2D log search BBME algorithm

    Args:
        previous (np.ndarray): previous frame
        current (np.ndarray): current frame
        mf (np.ndarray): motion field to compute
        height (int): rows of the frame
        width (int): columns of the frame
        pnorm_distance (int, optional): index of the dfd function to use. Defaults to 0 (mae)
        block_size (int, optional): size of each block (in pixels). Defaults to 4.
        search_window (int, optional): sized of the search window (in pixels). Defaults to 12.

    Returns:
        np.ndarray: motion field
    """

    positions = []
    for (block_row, block_col) in itertools.product(
        range(0, height - (block_size - 1), block_size),
        range(0, width - (block_size - 1), block_size),
    ):

        dx, dy = 0, 0
        step_size = search_window

        anchor_block = previous[
            block_row : block_row + block_size, block_col : block_col + block_size
        ]

        # get coordinates of current block
        x, y = block_row, block_col

        while step_size > 1:
            # initialize minimum
            min_block = np.infty

            # recompute positions to search
            positions.clear()
            if step_size > 2:
                # search positions are cross-shaped
                positions.append([x, y])
                positions.append([x + step_size, y])
                positions.append([x - step_size, y])
                positions.append([x, y + step_size])
                positions.append([x, y - step_size])
            elif step_size == 2:
                # search all the 8 positions in the neighborhood
                positions = list(
                    itertools.product([x - 2, x, x + 2], [y - 2, y, y + 2])
                )

            for (window_row, window_col) in positions:
                top_left, bottom_right = compute_current_target_block_corners(
                    x, y, window_row, window_col, block_size
                )

                if (
                    top_left[0] < 0
                    or top_left[1] < 0
                    or bottom_right[0] > height - 1
                    or bottom_right[1] > width - 1
                ):
                    continue

                current_block = current[
                    top_left[0] : bottom_right[0] + 1, top_left[1] : bottom_right[1] + 1
                ]
                dfd = compute_dfd(current_block, anchor_block, pnorm_function)

                if dfd < min_block:
                    min_block = dfd
                    dx = window_row
                    dy = window_col

            if dx == x and dy == y or step_size == 2:
                # update step size
                step_size //= 2

            # update coordinates
            x, y = dx, dy

        mf[block_row // block_size, block_col // block_size, 1] = dx - block_row
        mf[block_row // block_size, block_col // block_size, 0] = dy - block_col

    return mf


def diamond_search(
    previous,
    current,
    mf,
    height,
    width,
    pnorm_distance=0,
    block_size=12,
    search_window=-1,
):
    """
    diamond_search
    Computes the motion field with the diamond search BBME algorithm

    Args:
        previous (np.ndarray): previous frame
        current (np.ndarray): current frame
        mf (np.ndarray): motion field to compute
        height (int): rows of the frame
        width (int): columns of the frame
        pnorm_distance (int, optional): index of the dfd function to use. Defaults to 0 (mae)
        block_size (int, optional): size of each block (in pixels). Defaults to 4.

    Returns:
        nd.ndarray: motion field
    """

    large_search_pattern_offsets = [
        (0, 0),
        (2, 0),
        (1, 1),
        (0, 2),
        (-1, 1),
        (-2, 0),
        (-1, -1),
        (0, -2),
        (1, -1),
    ]
    small_search_pattern_offsets = [
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
    ]

    # notice here the approach at borders, at the moment we neglect right and bottom leftovers
    for (row, col) in itertools.product(
        range(0, height - block_size + 1, block_size),
        range(0, width - block_size + 1, block_size),
    ):
        # iterating over all image positions

        anchor_block = previous[row : row + block_size, col : col + block_size]
        match_position = (row, col)

        # large diamond search
        stopping_condition = False
        while not stopping_condition:
            min_diff = np.infty
            best_pos = match_position
            for offset in large_search_pattern_offsets:
                (row2, col2) = (
                    match_position[0] + offset[0],
                    match_position[1] + offset[1],
                )
                # wrap around a try catch block
                row2 = min(max(row2, 0), height - block_size - 1)
                col2 = min(max(col2, 0), width - block_size - 1)
                block = current[row2 : row2 + block_size, col2 : col2 + block_size]
                diff = compute_dfd(anchor_block, block, pnorm_distance)

                if diff < min_diff:
                    min_diff = diff
                    best_pos = (row2, col2)

            stopping_condition = match_position == best_pos
            match_position = best_pos

        # small diamond search
        min_diff = np.infty
        for offset in small_search_pattern_offsets:
            (row2, col2) = (
                match_position[0] + offset[1],
                match_position[1] + offset[0],
            )
            row2 = min(max(row2, 0), height - block_size - 1)
            col2 = min(max(col2, 0), width - block_size - 1)
            block = current[row2 : row2 + block_size, col2 : col2 + block_size]
            diff = compute_dfd(anchor_block, block, pnorm_distance)

            if diff < min_diff:
                min_diff = diff
                best_pos = (row2, col2)

        mf[row // block_size, col // block_size, 1] = best_pos[0] - row
        mf[row // block_size, col // block_size, 0] = best_pos[1] - col

    return mf


def rescale_motion_field(motion_field, scale=2):
    new_shape = (motion_field.shape[0] * scale, motion_field.shape[1] * scale, 2)
    mf = np.zeros(new_shape, dtype=np.int32)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            x = i // scale
            y = j // scale
            mf[i, j] = motion_field[x, y]
    mf = mf * 2
    return mf


def hierarchical_threestep(
    previous,
    current,
    block_size=10,
    search_window=4,
):
    """Hierarchical search implemented as threestep search.
    Computes the motion field with the threestep search in a hierarchical fashion.

    Args:
        previous (np.ndarray): previous frame.
        current (np.ndarray): current frame.
        block_size (int, optional): size of each block (in pixels).
        search_window (int, optional): size of the search window.

    Returns:
        nd.ndarray: motion field.
    """
    # compute pyramids from two frames
    previous_pyr = get_pyramids(previous, levels=3)
    current_pyr = get_pyramids(current, levels=3)

    # get first estimation for motion filed
    motion_field = get_motion_field(
        previous_pyr[0],
        current_pyr[0],
        block_size=block_size,
        searching_procedure=3,
        search_window=search_window,
    )

    for level in range(1, len(previous_pyr)):
        prev = previous_pyr[level]
        curr = current_pyr[level]
        # rescale motion field from previous level
        motion_field = rescale_motion_field(motion_field, scale=2)
        motion_field
        # get current level motion filed
        new_mf = get_motion_field(
            prev,
            curr,
            block_size=block_size,
            searching_procedure=3,
            search_window=search_window,
        )
        # solving problems with integer rounding
        if not motion_field.shape == new_mf.shape:
            if not motion_field.shape[0] == new_mf.shape[0]:
                filler = np.zeros((1, motion_field.shape[1], 2), dtype=np.int32)
                motion_field = np.vstack([motion_field, filler])
            else:
                filler = np.zeros((motion_field.shape[0], 1, 2), dtype=np.int32)
                motion_field = np.hstack([motion_field, filler])
        # smoothe current motion field
        motion_field = (motion_field + new_mf) / 2
    return motion_field


pnorm_distances = [mae, mse]
searching_procedures = [
    exhaustive_search,
    threestep_search,
    twodlog_search,
    diamond_search,
]


def main(args):
    frames = get_video_frames(args.path)

    previous = frames[args.fi - 3]
    current = frames[args.fi]

    motion_field = get_motion_field(
        previous,
        current,
        block_size=args.block_size,
        searching_procedure=args.searching_procedure,
        search_window=args.search_window,
    )
    motion_field_hierarchical = hierarchical_threestep(
        previous,
        current,
        block_size=args.block_size,
        search_window=args.search_window,
    )
    # print(motion_field)

    draw = draw_motion_field(previous, motion_field_hierarchical)
    cv2.imwrite(os.path.join("resources/images/motion_field_hierarchical.png"), draw)
    draw = draw_motion_field(previous, motion_field)
    cv2.imwrite(os.path.join("resources/images/motion_field_threestep.png"), draw)


if __name__ == "__main__":
    """
    example:
        python .\global_motion_estimation\bbme.py -p '.\videos\Faster-Animation480.m4v' -fi 8 -bs 4 -sp 2
    """
    parser = argparse.ArgumentParser(
        description="Computes motion field between two frames using block matching algorithms"
    )
    parser.add_argument(
        "-p",
        "--video-path",
        dest="path",
        type=str,
        required=True,
        help="path of the video to analyze",
    )
    parser.add_argument(
        "-fi",
        "--frame-index",
        dest="fi",
        type=int,
        required=True,
        help="index of the current frame to analyze in the video",
    )
    parser.add_argument(
        "-pn",
        "--p-norm",
        dest="pnorm",
        type=int,
        default=0,
        help="pnorm distance to use",
    )
    parser.add_argument(
        "-bs",
        "--block-size",
        dest="block_size",
        type=int,
        default=12,
        help="size of the block",
    )
    parser.add_argument(
        "-sw",
        "--search-window",
        dest="search_window",
        type=int,
        default=8,
        help="size of the search window",
    )
    parser.add_argument(
        "-sp",
        "--searching-procedure",
        dest="searching_procedure",
        type=int,
        default=1,
        help="0: Exhaustive search,\n"
        "1: Three Step search,\n"
        "2: 2D Log search,\n"
        "3: Diamond search",
    )
    args = parser.parse_args()

    main(args)
