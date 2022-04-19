from utils import get_video_frames, draw_motion_field, PSNR
import motion as motion
from json import dump
import numpy as np
import argparse
import shutil
import cv2
import os


FRAME_DISTANCE = 1


def main(args):

    video = args.path
    if args.fd is not None:
        FRAME_DISTANCE = int(args.fd)

    video_path = os.path.join("resources", "videos", video)
    results_path = os.path.join("results", "")
    save_path = os.path.join(results_path, video.replace(".mp4", ""), "")
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    os.mkdir(save_path)
    os.mkdir(os.path.join(save_path, "frames", ""))
    os.mkdir(os.path.join(save_path, "compensated", ""))
    os.mkdir(os.path.join(save_path, "curr_prev_diff", ""))
    os.mkdir(os.path.join(save_path, "model_motion_field", ""))
    os.mkdir(os.path.join(save_path, "curr_comp_diff", ""))

    psnr_dict = {}
    frames = get_video_frames(video_path)
    try:
        print("frame shape: {}".format(frames[0].shape))
    except:
        raise Exception("Error reading video file: check the name of the video!")
    for idx in range(FRAME_DISTANCE, len(frames)):

        j = (idx + 1) / len(frames)
        print("[%-20s] %d/%d frames" % ("=" * int(20 * j), idx, len(frames)))

        # get previous, current and compensated
        previous = frames[idx - FRAME_DISTANCE]
        current = frames[idx]

        params = motion.global_motion_estimation(previous, current)

        model_motion_field = motion.get_motion_field_affine(
            (int(previous.shape[0] / motion.BBME_BLOCK_SIZE), int(previous.shape[1] / motion.BBME_BLOCK_SIZE), 2), parameters=params
        )


        shape = (previous.shape[0]//motion.BBME_BLOCK_SIZE, previous.shape[1]//motion.BBME_BLOCK_SIZE)
        # compensate camera motion on previous frame
        compensated = motion.compensate_frame(previous, model_motion_field)


        idx_name = str(idx)
        # save frames
        cv2.imwrite(
            os.path.join(save_path, "frames", "")
            + str(idx-5)
            + ".png",
            previous,
        )

        # save compensated
        cv2.imwrite(
            os.path.join(save_path, "compensated", "")
            + str(idx-5)
            + ".png",
            compensated,
        )

        # save differences
        diff_curr_prev = (
            np.absolute(current.astype("int") - previous.astype("int"))
        ).astype("uint8")
        diff_curr_comp = (
            np.absolute(current.astype("int") - compensated.astype("int"))
        ).astype("uint8")
        cv2.imwrite(
            os.path.join(save_path, "curr_prev_diff", "")
            + str(idx)
            + ".png",
            diff_curr_prev,
        )
        cv2.imwrite(
            os.path.join(save_path, "curr_comp_diff", "")
            + str(idx)
            + ".png",
            diff_curr_comp,
        )

        # save motion model motion field
        draw = draw_motion_field(previous, model_motion_field)
        cv2.imwrite(
            os.path.join(save_path, "model_motion_field", "")
            + str(idx)
            + ".png",
            draw,
        )

        ## compute PSNR
        psnr = PSNR(current, compensated)
        psnr_dict[idx_name] = str(psnr)
        with open(save_path + "psnr_records.json", "w") as outfile:
            dump(psnr_dict, outfile)

if __name__ == "__main__":
    """Once set the video path and save path creates a lot of data for your report. Namely, it saves frames, compensated frames, frame differences and estimations of global motion.
    """
    parser = argparse.ArgumentParser(
        description="Launches GME and yields results"
    )
    parser.add_argument(
        "-v",
        "--video-name",
        dest="path",
        type=str,
        required=True,
        help="name of the video to analyze (no ext)",
    )
    parser.add_argument(
        "-f",
        "--frame-distance",
        dest="fd",
        type=str,
        required=False,
        help="frame displacement",
    )
    args = parser.parse_args()

    main(args)



    