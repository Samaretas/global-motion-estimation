from motion import dense_motion_estimation
from utils import get_video_frames
import cv2

frames = get_video_frames("./hall_objects_qcif.y4m")
idx = 30

precedent = frames[idx-1]
current = frames[idx]

cv2.imshow("cur", current)
cv2.waitKey(0)

motion_field = dense_motion_estimation(precedent, current)
print("motion_field")
print(type(motion_field))
print(motion_field.shape)
print(motion_field.max)
print(motion_field)

print("end")