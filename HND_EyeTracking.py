'''
This is a module built with the intention of tracking eye movement for the LEAP project - NIRS videocoding.
We attempt to do so through the utilization of the dlib and OpenCV-Python modules, which track objects through videos.
'''

import cv2 as cv
import dlib
import numpy as np
import pandas as pd
import sys
import math
from AudioDetection import blockStart
from EPrimeExtract import blocks
from InitialCoordinates import coordinates


# face detector object
detectFace = dlib.get_frontal_face_detector()
# additional face detector; need to have "haarcascade_frontalface_default.xml" file in same directory
cascPath = "haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier(cascPath)

# landmarks detector; need to have "shape_predictor_68_face_landmarks.dat" file in same directory
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# open the video file; modify run configuration to input the path to any video file
camera = cv.VideoCapture(sys.argv[1])


def faceDetector(image, gray, prevFace):

    # getting faces from face detector
    faces = detectFace(gray)
    faces2 = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    # initializing face object
    face = None
    if prevFace is not None:
        prevCenter = ((prevFace.left() + prevFace.right()) / 2, (prevFace.top() + prevFace.bottom()) / 2)
    # looping through all detected faces
    for face in faces:
        # getting coordinates of face
        cordFace1 = (face.left(), face.top())
        cordFace2 = (face.right(), face.bottom())
        center = ((face.left() + face.right()) / 2, (face.top() + face.bottom()) / 2)
        if prevFace is None:
            if math.sqrt((center[0] - coordinates[0]) ** 2 + (center[1] - coordinates[1]) ** 2) > 25:
                face = None
                continue
        else:
            if math.sqrt((center[0] - prevCenter[0]) ** 2 + (center[1] - prevCenter[1]) ** 2) > 50:
                face = None
                continue
        cv.rectangle(image, cordFace1, cordFace2, (0, 255, 0), 2)
        return image, face

    for (x, y, w, h) in faces2:
        # since this is the less-accurate method, checking previous face coordinates are within range
        if prevFace is not None:
            center = ((2*x + h) / 2, (2*y + h) / 2)
            if math.sqrt((center[0] - prevCenter[0]) ** 2 + (center[1] - prevCenter[1]) ** 2) > 25:
                continue
            face = dlib.rectangle(x, y, x + h, y + h)
            cv.rectangle(image, (x, y), (x + h, y + h), (255, 0, 0), 2)
        break
    return image, face


def click_event(event, x, y, flags, params):

    # function used to click on middle of child's face in case of movement
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        # change lastFace to None then act as though we are extracting new coordinates from the first frame
        global lastFace
        lastFace = None
        coordinates[0] = x
        coordinates[1] = y


look = ""
looks = []
lookTimes = []
behType = ""
types = []
times = []
trialNum = 0
trialCh = False
trialNumList = []
videoTime = 0
lastFace = None
# var for tracking time in between start of not looking and current time for detecting >5s
startnlTime = 0.0
# records which trial begins not looking
startTrial = 0
notLookingTime = {1: 0.0}
# composite dictionary of look, type, time, and trial: {time: [look, type, trialNum]}, used for later ordering times
lookDict = {}
# unique file name for each video
initials = input("\nPlease insert your initials and press Enter: ")

# set start time and difference between video file and E-Prime file
start = blocks[1][0][1]
diff = blockStart - start

while True:
    # getting frame from video
    ret, frame = camera.read()
    if not ret:
        break

    # only include the frames in blocks
    t = round(camera.get(cv.CAP_PROP_POS_MSEC) / 1000, 3)
    inBlock = False
    newTrial = False
    for trial, pair in blocks.items():
        if pair[0][0] <= t - diff <= pair[0][2]:
            inBlock = True
            times.append(t)
            if trial != trialNum:
                trialNum = trial
                newTrial = True
                trialCh = True
                notLookingTime[trialNum] = 0.0
                # append time point of new trial to spreadsheet
                if trialNum == 1:
                    lookDict[t] = ["Baseline", "POINT", trialNum]
                else:
                    lookDict[videoTime] = ["Baseline", "POINT", trialNum]
            # append trial type to file, only ensure 1 point of change
            if abs(t - diff - pair[0][1]) <= 1 / camera.get(cv.CAP_PROP_FPS) and trialCh:
                lookDict[pair[0][1] + diff] = [pair[1], "POINT",
                                               list(x for xs in list(lookDict.values()) for x in xs).count(pair[1]) + 1]
                trialCh = False
            break
    if not inBlock:
        continue
    videoTime = round(t, 3)

    # adjusting brightness/contrast of frame
    cv.normalize(frame, frame, 0, 255, cv.NORM_MINMAX)
    if sys.argv[1].split("\\")[4] == "1yo":
        # alpha = contrast; beta = brightness; can be adjusted on a video-by-video basis
        frame = cv.convertScaleAbs(frame, alpha=0.9, beta=30)
    else:
        B, G, R = cv.split(frame)
        B = cv.equalizeHist(B)
        G = cv.equalizeHist(G)
        R = cv.equalizeHist(R)
        frame = cv.merge((B, G, R))
    # adjusting sharpness of the frame
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    frame = cv.filter2D(frame, -1, kernel)

    # converting frame into gray image
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calling the face detector function
    image, face = faceDetector(frame, grayFrame, lastFace)

    if face is None:
        # indicative that code is not picking up face --> child is not looking forward, is sleeping, or poor detection
        if look == "Looking" or look == "":
            look = "NotLooking"
            startnlTime = videoTime
            startTrial = trialNum
            behType = "START"

        # display plain frame
        cv.imshow('Frame', frame)

    else:
        lastFace = face
        # trialFace[trialNum] += 1
        if look == "NotLooking":
            # only record not looking for more than 5 seconds, shorter may be loss of tracking for brief time by program
            if videoTime - startnlTime > 5:
                # append start and stop times
                lookDict[startnlTime + 0.001] = [look, "START", ""]
                if trialNum == startTrial:
                    notLookingTime[trialNum] += videoTime - startnlTime
                # consecutive trials
                elif (trialNum - startTrial) < 2:
                    notLookingTime[startTrial] += (blocks[startTrial][0][2] + diff) - startnlTime
                    notLookingTime[trialNum] += videoTime - (blocks[trialNum][0][0] + diff)
                # more than 2 trials of not looking
                else:
                    notLookingTime[startTrial] += (blocks[startTrial][0][2] + diff) - startnlTime
                    for trials in range(startTrial + 1, trialNum):
                        notLookingTime[trials] += blocks[trials][0][2] - blocks[trials][0][0]
                    notLookingTime[trialNum] += videoTime - (blocks[trialNum][0][0] + diff)
                lookDict[videoTime + 0.001] = [look, "STOP", ""]
            look = "Looking"
        # display frame detecting face
        cv.imshow('Frame', image)

    # adjust child coordinates if big move occurs while face isn't tracked
    cv.setMouseCallback('Frame', click_event)
    # if q is pressed on keyboard: quit
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# append stop if not looking and final baseline
if look == "NotLooking":
    if videoTime - startnlTime > 5:
        lookDict[videoTime] = [look, "STOP", ""]
        if trialNum == startTrial:
            notLookingTime[trialNum] += videoTime - startnlTime
        # consecutive trials
        elif trialNum - startTrial < 2:
            notLookingTime[startTrial] += (blocks[startTrial][0][2] + diff) - startnlTime
            notLookingTime[trialNum] += videoTime - (blocks[trialNum][0][0] + diff)
        # more than 2 trials of not looking
        else:
            notLookingTime[startTrial] += (blocks[startTrial][0][2] + diff) - startnlTime
            for trials in range(startTrial + 1, trialNum):
                notLookingTime[trials] += blocks[trials][0][2] - blocks[trials][0][0]
            notLookingTime[trialNum] += videoTime - (blocks[trialNum][0][0] + diff)
lookDict[videoTime + 0.001] = ["End", "POINT", trialNum + 1]

# order dictionary to display in Excel file at correct times
ordered = dict(sorted(lookDict.items(), key=lambda item: item[0]))
obsID = [sys.argv[1].split("\\")[-1]] * len(list(ordered.keys()))
for key, val in list(ordered.items()):
    looks.append(val[0])
    types.append(val[1])
    trialNumList.append(val[2])
    lookTimes.append(key)

# checking percentage of valid footage for each block
validData = {}
for trial, nlTime in notLookingTime.items():
    validData[trial] = 100 - round(nlTime / (blocks[trial][0][2] - blocks[trial][0][0]) * 100, 1)
# insert % of child looking for each block
percLooking = [np.nan] * len(list(lookDict.keys()))
timeLooking = [np.nan] * len(list(lookDict.keys()))
for index, val in enumerate(list(ordered.values())):
    if val[2] != "":
        if val[0] == "Familiar":
            percLooking[index] = validData[val[2]]
            timeLooking[index] = blocks[val[2]][0][2] - notLookingTime[val[2]]
        elif val[0] == "Novel":
            percLooking[index] = validData[val[2] + 15]
            timeLooking[index] = blocks[val[2] + 15][0][2] - notLookingTime[val[2] + 15]
        elif val[0] == "Repeat":
            percLooking[index] = validData[val[2] + 20]
            timeLooking[index] = blocks[val[2] + 20][0][2] - notLookingTime[val[2] + 20]
# add how many trials under 60% looking to file name (i.e., how many manual coders need to code)
lessThan60 = 0
for perc in percLooking:
    if perc != np.nan:
        if float(perc) < 60:
            lessThan60 += 1

# add times in which child is not looking to Excel file
fileName = (sys.argv[1].split("\\")[-1].split("_")[0][:3] + "_" + sys.argv[1].split("\\")[-1].split("_")[0][3:] + "_" +
            sys.argv[1].split("\\")[-1].split("_")[1] + "_HND_auto_" + initials + "_" + str(lessThan60) + ".xlsx")
df = pd.DataFrame({"Observation id": obsID, "Behavior": looks, "Behavior Type": types, "Modifier #1": trialNumList,
                   "Time": lookTimes, "Looking Time": timeLooking, "Looking %": percLooking})
df.to_excel(fileName, sheet_name=sys.argv[1].split("\\")[-1], index=False)

# releasing video capture
camera.release()
cv.destroyAllWindows()
