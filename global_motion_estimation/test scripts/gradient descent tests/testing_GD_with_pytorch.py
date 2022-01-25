from utils import get_video_frames
from global_motion_estimation.motion import global_motion_estimation, first_estimation
import cv2
import numpy as np
from gradient_descent_pytorch import *

frames = get_video_frames("./hall_objects_qcif.y4m")

for idx in range(30, len(frames)):
    previous = frames[idx-1]
    current = frames[idx]

    # cv2.imshow("cur", current)
    # cv2.waitKey(1)

    # global_motion_estimation(previous, current)
    parameters = first_estimation(previous, current)    # to test first estimation of the parameters
    
    weights = torch.DoubleTensor(parameters)
    # instantiate model 
    m = Model(weights)
    print(m.weights)
    # Instantiate optimizer 
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    losses = training_loop(m, opt, previous, current)
    plt.figure(figsize=(14, 7)) 
    plt.plot(losses) 
    print(m.weights)
    break

print("pytorch GD tested")

#! it does not work
"""
    Why? well, there could be several reasons, the most important: we are not using backprop in the standard fashion.
    In the first place, y predicted for us is the compensated frame, where y target is the current frame, but this is not standard in the literature.
    Then I sincerely don't know how the flow of pytorch goes in this case, therefore, it's possible that I inserted some dummy-error around.
"""