import cv2

def get_video_frames(path):
    cap = cv2.VideoCapture(path)
    flag = True
    frames = list()
    while flag:
        if cap.grab():
            flag, frame = cap.retrieve()
            frames.append(frame)
        else:
            flag=False
    return frames

def get_pyramids(original_image, levels=3):
    pyramid = [original_image]
    curr = original_image
    for i in range(1, levels):
        scaled = cv2.pyrDown(curr)
        curr = scaled
        pyramid.insert(0, scaled)
    return pyramid


if __name__ == "__main__":
    import time

    # frames = get_video_frames("./hall_objects_qcif.y4m")
    # for frame in frames:            
    #     cv2.imshow('video', frame)
    #     cv2.waitKey(1)

    # pyr = get_pyramids(frames[0], 3)
    # for level in pyr:
    #     cv2.imshow('lvl', level)
    #     cv2.waitKey(1000)

