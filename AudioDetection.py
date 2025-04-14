'''
This module is designed to complement VWM_EyeTracking.py, HND_EyeTracking.py by allowing audio and visual
representations of the video in real speed to allow the user to press the "Enter" key to signal when the eye tracking of
the video should begin. User should hit the enter key once the see or hear the initial "star sound" OR block flash to
allow for proper measurement in the corresponding file.
'''

import cv2 as cv
import ffpyplayer.player as pl
import sys
import time
from EPrimeExtract import trialType


blockStart = 0
startSig = ""

while True:
    # open the video and audio file
    camera = cv.VideoCapture(sys.argv[1])
    audio = pl.MediaPlayer(sys.argv[1])
    start = time.time()
    if trialType == "VWM":
        print("\nPress Enter at Baseline (star-sound or first set of blocks).")
    elif trialType == "HND":
        print("\nPress Enter at Baseline (when first trial sentence is spoken).")

    while True:
        # getting frame from video
        ret, frame = camera.read()
        if not ret:
            break
        # getting audio from video
        audio_frame, val = audio.get_frame()
        if val != 'eof' and audio_frame is not None:
            img, t = audio_frame
        cv.imshow("Video", frame)

        elapsed = (time.time() - start) * 1000
        play_time = int(camera.get(cv.CAP_PROP_POS_MSEC))
        sleep = max(1, int(play_time - elapsed))
        # press "Enter" or "Return" key to signal time of beginning-block signal
        if cv.waitKey(sleep) & 0xFF == ord("\r"):
            blockStart = round(camera.get(cv.CAP_PROP_POS_MSEC) / 1000, 3)
            break

    camera.release()
    cv.destroyAllWindows()
    audio.set_pause(True)

    while True:
        if trialType == "VWM":
            # ask user whether they used the key to signal star sound or block flash
            ans = input("\nDid you press Enter when the (1) star sound initiated or when the (2) blocks flashed?\n"
                        "Type \"1\" or \"2\" then press Enter. If you did not press Enter at a correct time, "
                        "type \"3\". ")
        else:
            # ask if trial start time was correct
            ans = input("\nDid you press Enter at the correct time? Type \"Y\" or \"N\" then press Enter. ")
        if ans in ["1", "2", "3", "Y", "N"]:
            if ans == "Y":
                startSig = "2"
            else:
                startSig = ans
            break
        else:
            print("Invalid Input.")
    # break out of audio file if correct start time identified
    if startSig == "1" or startSig == "2":
        break
