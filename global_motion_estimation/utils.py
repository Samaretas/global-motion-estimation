from cmath import log10, sqrt
from tkinter import image_names
import cv2
import time
import json
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
    frame_dummy = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    
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
                # (120, 120, 120),
                (0, 0, 255),
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
    for i in range(3, num_frames):
        s = str(i-3)+"-"+str(i)+".png"
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

def some_data(psnr_path: str) -> None:
    psnrs = {}
    with open(psnr_path, 'r') as f:
        psnrs = json.load(f)

    psnrs_np = np.zeros(shape=[len(psnrs),1])
    count = 0
    for frames in psnrs:
        cut = psnrs[frames].index('+')
        num = psnrs[frames][1:cut]
        psnrs_np[count] = num
        count += 1

    avg = psnrs_np.sum() / len(psnrs)
    diff = np.zeros(shape=[len(psnrs),1])

    for value in psnrs_np:
        idx = psnrs_np.tolist().index(value) 
        diff[idx] = (value - avg) ** 2

    var = (diff.sum()/len(psnrs))

    print("Average: {:.3f}".format(avg))
    print("Variance: {:.3f}".format(var))
    print("Standard deviation: {:.3f}".format(var**(1/2)))
    print("Highest: {:.3f}".format(psnrs_np.max()))
    print("Lowest: {:.3f}".format(psnrs_np.min()))




if __name__ == "__main__":
    # create_video_from_frames("./results/mat_inv_nastro3/bbme/", 75, "pan_nastro_3_bbme.avi", 5)
    # create_video_from_frames("./results/mike_ball/gme/", 80, "mike_bounce.avi", 20)
    # create_video_from_frames("./results/pan240/diff_curr_comp/", 205, "pan240_compensated_gme.avi", 30)
    # create_video_from_frames("./results/pan240/diff_curr_prev/", 205, "pan240_not_compensated.avi", 30)
    # matrix = np.arange(25)
    # matrix = matrix.reshape((5,5))
    # print(matrix)
    # matrix = matrix.flatten()
    # print(matrix)
    # histogram = np.histogram(matrix, np.array([i for i in range(10)]))
    # print(histogram)
    # some_data('venv/psnr_records.json')
    import os
    base_path = os.path.join("results")
    for d in os.listdir(base_path):
        video_data = os.path.join(base_path, d, "psnr_records.json")
        print(f"video {d}") # name of video
        some_data(video_data)
        print("======================")