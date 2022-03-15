from cmath import log10, sqrt
from tkinter import image_names
import cv2
import time
import numpy as np


def get_video_frames(path):
    """
    Given the path of the video capture, returns the list of frames.
    Frames are converted in grayscale.

    Argss:
        path (str): path to the video capture

    Returns:
        frames (list):  list of grayscale frames of the specified video
    """
    cap = cv2.VideoCapture(path)
    flag = True
    frames = list()
    while flag:
        if cap.grab():
            flag, frame = cap.retrieve()
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        else:
            flag = False
    return frames


def get_pyramids(original_image, levels=3):
    """
    Rturns a list of downsampled images, obtained with the Gaussian pyramid method. The length of the list corresponds to the number of levels selected.

    Args:
        original_image (np.ndarray): the image to build the pyramid with
        levels (int): the number of levels (downsampling steps), default to 3

    Returns:
        pyramid (list): the listwith the various levels of the gaussian pyramid of the image.
    """
    pyramid = [original_image]
    curr = original_image
    for i in range(1, levels):
        scaled = cv2.pyrDown(curr)
        curr = scaled
        pyramid.insert(0, scaled)
    return pyramid


def draw_motion_field(frame, motion_field):
    height, width = frame.shape
    frame_dummy = frame.copy()
    mf_height, mf_width, _ = motion_field.shape
    bs = height // mf_height

    for y in range(0, mf_height):
        for x in range(0, mf_width):
            idx_x = x * bs + bs//2
            idx_y = y * bs + bs//2
            mv_x, mv_y = motion_field[y][x]

            cv2.arrowedLine(
                frame_dummy,
                (idx_x, idx_y),
                (int(idx_x + mv_x), int(idx_y + mv_y)),
                (120, 120, 120),
                1,
                line_type=cv2.LINE_AA
            )
    return frame_dummy


def timer(func):
    """
    Decorator that prints the time of execution of a certain function.

    Args:
        func (Callable[[Callable], Callable]): the function that has to be decorated (timed)

    Returns:
        wrapper (Callable[[any], any]): the decorated function
    """

    def wrapper(*args, **kwargs):
        start = int(time.time())
        ret = func(*args, **kwargs)
        end = int(time.time())
        print(f"Execution of '{func.__name__}' in {end-start}s")
        return ret

    return wrapper


def PSNR(original, noisy):
    """
    Computes the peak sognal to noise ratio.

    Args:
        original (np.ndarray): original image
        noisy (np.ndarray): noisy image

    Returns:
        float: the measure of PSNR
    """
    mse = np.mean((original.astype("int") - noisy.astype("int")) ** 2)
    if mse == 0:  # there is no noise
        return -1
    max_value = 255.0
    psnr = 20 * log10(max_value / sqrt(mse))
    return psnr


def create_video_from_frames(frame_path, num_frames, video_name, fps=30):
    import os
    img_array = []
    img_names = []
    for i in range(num_frames):
        s = str(i)+"-"+str(i+1)+".png"
        img_names.append(s)
    for img in img_names:
        image = cv2.imread(frame_path+img)
        img_array.append(image)
    height, width, layers = img_array[0].shape
    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    for image in img_array:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    # create_video_from_frames("./results/mat_inv_nastro3/bbme/", 75, "pan_nastro_3_bbme.avi", 5)
    # create_video_from_frames("./results/mat_inv_nastro3/gme/", 75, "pan_nastro_3_gme_mat_inv.avi", 5)
    matrix = np.arange(25)
    matrix = matrix.reshape((5,5))
    print(matrix)
    matrix = matrix.flatten()
    print(matrix)
    histogram = np.histogram(matrix, np.array([i for i in range(10)]))
    print(histogram)