from utils import get_video_frames, get_pyramids
from bbme import Block_matcher

frames = get_video_frames("./hall_objects_qcif.y4m")

for idx in range(1, len(frames)):
    precedent = frames[idx-1]
    current = frames[idx]

    precedent_pyr = get_pyramids(precedent)
    current_pyr = get_pyramids(current)

    # buggy code down here

    BM = Block_matcher(block_size=6,
                         search_range=2,
                         pixel_acc=1,
                         searching_procedure = 2)

    _, motion_field = BM.get_motion_field(precedent_pyr[0], current_pyr[0])
    print(motion_field)
    exit()