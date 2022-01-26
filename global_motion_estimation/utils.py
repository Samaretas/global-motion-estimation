import cv2
import time

def get_video_frames(path):
    """
        Given the path of the video capture, returns the list of frames.
        Frames are converted in grayscale.

        Prameters:
        @path   path to the video capture
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
            flag=False
    return frames

def get_pyramids(original_image, levels=3):
    """
        Rturns a list of downsampled images, obtained with the Gaussian pyramid method. The length of the list corresponds to the number of levels selected.

        @original_image     the image to build the pyramid with
        @levels             the number of levels (downsampling steps), default to 3
    """
    pyramid = [original_image]
    curr = original_image
    for i in range(1, levels):
        scaled = cv2.pyrDown(curr)
        curr = scaled
        pyramid.insert(0, scaled)
    return pyramid

def draw_motion_vector(frame, motion_field):
    height, width = frame.shape
    frame_dummy = frame.copy()
    mv_h , mv_w,_ = motion_field.shape
    b_size = int(height/mv_h)

    for y in range(0, mv_h ):
        for x in range(0, mv_w ):
            idx_x = x * b_size
            idx_y = y * b_size
            mv_x, mv_y = motion_field[y][x]

            cv2.arrowedLine(frame_dummy, (idx_x, idx_y), (int(idx_x + mv_x), int(idx_y + mv_y)), (0, 255, 0), 1)
    return frame_dummy

def timer(func):
    def wrapper(*args, **kwargs):
        start = int(time.time())
        ret = func(*args, **kwargs)
        end = int(time.time())
        print(f"Execution of function {func.__name__} in {end-start}s")
        return ret
    return wrapper


if __name__ == "__main__":
    frames = get_video_frames("./hall_objects_qcif.y4m")
    pyrs = get_pyramids(frames[30])
    for pyramid in pyrs:
        cv2.imshow("Pyr", pyramid)
        cv2.waitKey(0)