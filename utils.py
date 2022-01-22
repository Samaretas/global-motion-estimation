import cv2

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


if __name__ == "__main__":
    frames = get_video_frames("./hall_objects_qcif.y4m")
    pyrs = get_pyramids(frames[30])
    for pyramid in pyrs:
        cv2.imshow("Pyr", pyramid)
        cv2.waitKey(0)