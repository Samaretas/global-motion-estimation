from utils import get_video_frames, PSNR
import motion as motion
from json import dump
import numpy as np
import argparse
import cv2
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-vp", "--video_path", help="path to the video you want to analyze", type=str
    )
    parser.add_argument(
        "-sp",
        "--save_path",
        help="path to folder where to save all the results",
        type=str,
    )
    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        try:
            os.mkdir(args.save_path)
        except:
            raise Exception("The save path specified is not a folder")
    if not os.path.isdir(args.save_path + "compensated"):
        os.mkdir(args.save_path + "compensated")
        os.mkdir(args.save_path + "diff_comp_curr")
        os.mkdir(args.save_path + "diff_comp_prev")
        os.mkdir(args.save_path + "diff_curr_prev")
    frames = get_video_frames(args.video_path)
    psnr_dict = {}

    for idx in range(1, len(frames)):
        j = (idx + 1) / len(frames)
        print("[%-20s] %d/%d frames" % ('='*int(20*j), idx, len(frames)))
        previous = frames[idx - 1]
        current = frames[idx]

        compensated = motion.global_motion_estimation(previous, current)
        diff_comp_curr = (
            np.absolute(compensated.astype("int") - current.astype("int"))
        ).astype("uint8")
        diff_comp_prev = (
            np.absolute(compensated.astype("int") - previous.astype("int"))
        ).astype("uint8")
        diff_curr_prev = (
            np.absolute(current.astype("int") - previous.astype("int"))
        ).astype("uint8")
        idx_name = str(idx - 1) + "-" + str(idx)
        cv2.imwrite(args.save_path + "compensated\\" + idx_name + ".png", compensated)
        cv2.imwrite(
            args.save_path + "diff_comp_curr\\" + idx_name + ".png", diff_comp_curr
        )
        cv2.imwrite(
            args.save_path + "diff_comp_prev\\" + idx_name + ".png", diff_comp_prev
        )
        cv2.imwrite(
            args.save_path + "diff_curr_prev\\" + idx_name + ".png", diff_curr_prev
        )

        psnr = PSNR(current, compensated)
        psnr_dict[idx_name] = str(psnr)
    with open(args.save_path + "psnr_records.json", "w") as outfile:
        dump(psnr_dict, outfile)