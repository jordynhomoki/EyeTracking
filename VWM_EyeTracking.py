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
import time
from AudioDetection import blockStart, startSig
from EPrimeExtract import blocks, yo
from InitialCoordinates import coordinates


# compute amount of time it takes to complete video coding
runtime_start = time.time()

# face detector object
detectFace = dlib.get_frontal_face_detector()
# additional face detector; need to have "haarcascade_frontalface_default.xml" file in same directory
cascPath = "haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier(cascPath)

# landmarks detector; need to have "shape_predictor_68_face_landmarks.dat" file in same directory
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# open the video file; modify run configuration to input the path to any video file
camera = cv.VideoCapture(sys.argv[1])


def faceDetector(image, gray, prevFace, newBlock):

    # getting faces from face detectors
    faces = detectFace(gray)
    faces2 = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
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
            if math.sqrt((center[0] - coordinates[0]) ** 2 + (center[1] - coordinates[1]) ** 2) > 10:
                face = None
                continue
        else:
            if not newBlock:
                if math.sqrt((center[0] - prevCenter[0]) ** 2 + (center[1] - prevCenter[1]) ** 2) > 25:
                    face = None
                    continue
            else:
                if math.sqrt((center[0] - prevCenter[0]) ** 2 + (center[1] - prevCenter[1]) ** 2) > 50:
                    face = None
                    continue
        cv.rectangle(image, cordFace1, cordFace2, GREEN, 2)
        return image, face

    for (x, y, w, h) in faces2:
        # since this is the less-accurate method, checking previous face coordinates are within range
        if prevFace is not None:
            center = ((2*x + h) / 2, (2*y + h) / 2)
            if not newBlock:
                if math.sqrt((center[0] - prevCenter[0]) ** 2 + (center[1] - prevCenter[1]) ** 2) > 5:
                    continue
            else:
                if math.sqrt((center[0] - prevCenter[0]) ** 2 + (center[1] - prevCenter[1]) ** 2) > 25:
                    continue
            face = dlib.rectangle(x, y, x + h, y + h)
            cv.rectangle(image, (x, y), (x + h, y + h), BLUE, 2)
            break
    return image, face


def faceLandmarkDetector(image, gray, face):

    # landmarks predictor
    landmarks = predictor(gray, face)
    pointList = []
    # looping through each landmark to append coordinates
    for n in range(0, 68):
        point = (landmarks.part(n).x, landmarks.part(n).y)
        pointList.append(point)
    return image, pointList


def EyeTracking(gray, eyePoints):

    dim = gray.shape
    mask = np.zeros(dim, dtype=np.uint8)
    # converting eyePoints into Numpy arrays.
    pollyPoints = np.array(eyePoints, dtype=np.int32)
    # filling the eyes with WHITE color to be distinguished from gray
    cv.fillPoly(mask, [pollyPoints], color=WHITE)
    # writing gray image where color is white in the mask
    eyeImage = cv.bitwise_and(gray, gray, mask=mask)

    # getting the max and min points of eye in order to crop the eyes from Eye image
    maxX = (max(eyePoints, key=lambda item: item[0]))[0]
    minX = (min(eyePoints, key=lambda item: item[0]))[0]
    maxY = (max(eyePoints, key=lambda item: item[1]))[1]
    minY = (min(eyePoints, key=lambda item: item[1]))[1]

    # where mask is 0 (everywhere besides eyes), make color black
    eyeImage[mask == 0] = 255

    # crop eyeImage
    croppedEye = eyeImage[minY:maxY, minX:maxX]
    height, width = croppedEye.shape

    # applying binary threshold to the eye
    ret, thresholdEye = cv.threshold(croppedEye, 100, 255, cv.THRESH_BINARY)
    if thresholdEye is None:
        return None

    # dividing the eye into right, center, and left parts
    # make center portion smaller or eliminate entirely ? --> higher right/left detection
    # make extreme corners (furthest 1/6) of eye not looking
    divPart = int(width / 6)
    rightPart = thresholdEye[0:height, divPart:3*divPart]
    # centerPart = thresholdEye[0:height, 3*divPart:4*divPart]
    leftPart = thresholdEye[0:height, 3*divPart:5*divPart]
    outsidePart1 = thresholdEye[0:height, 0:divPart]
    outsidePart2 = thresholdEye[0:height, 5*divPart:width]

    # counting the black pixels in each part of the eye
    rightBlackPx = np.sum(rightPart == 0)
    # centerBlackPx = np.sum(centerPart == 0)
    leftBlackPx = np.sum(leftPart == 0)
    outside1BlackPx = np.sum(outsidePart1 == 0)
    outside2BlackPx = np.sum(outsidePart2 == 0)
    # get position/direction of eyes based on maximal black pixels
    pos = Position([rightBlackPx, leftBlackPx, outside1BlackPx, outside2BlackPx])
    return pos


def Position(ValuesList):

    maxIndex = ValuesList.index(max(ValuesList))
    posEye = ""
    if maxIndex == 0:
        posEye = "Right"
    #elif maxIndex == 1:
    #    posEye = "Center"
    elif maxIndex == 1:
        posEye = "Left"
    elif maxIndex == 2 or maxIndex == 3:
        posEye = "Outside"
    else:
        posEye = "Closed"
    return posEye


def click_event(event, x, y, flags, params):

    # function used to click on middle of child's face in case of movement
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        # change lastFace to None then act as though we are extracting new coordinates from the first frame
        global lastFace
        lastFace = None
        coordinates[0] = x
        coordinates[1] = y


GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

left = {}
right = {}
look = ""
looks = []
lookTimes = []
behType = ""
types = []
videoTime = 0.0
start = 0
times = []
blockNum = 1
blockNumList = []
blockFace = {1: 0}
lastFace = None
frames = {1: 0}
notLookingTime = {1: 0.0}
# not looking boolean
nl = False
# unique file name for each video
initials = input("\nPlease insert your initials and press Enter: ")

# change start time and difference between video file and E-Prime file based on starting signal
if startSig == "1":
    start = list(blocks.keys())[0]
else:
    start = list(blocks.values())[0][0]
diff = blockStart - start
startnlTime = blockStart

while True:
    # getting frame from video
    ret, frame = camera.read()
    if not ret:
        break

    # only include the frames in blocks
    t = round(camera.get(cv.CAP_PROP_POS_MSEC) / 1000, 3)
    inBlock = False
    baseline = 0.0
    for base, pair in blocks.items():
        if pair[0] <= t - diff <= pair[1]:
            inBlock = True
            times.append(t)
            if blockNum == 1:
                baseline = round(base - 8.270, 3)
            else:
                # end of previous block
                baseline = times[-2] - diff
                # baseline = list(blocks.values())[blockNum - 1][1]
            break
    if not inBlock:
        continue
    # observe when video moves to a different block
    if t - times[len(times) - 2] > 10:
        blockNum += 1
        blockFace[blockNum] = 0
        frames[blockNum] = 0
    newBlock = False
    if frames[blockNum] == 0:
        newBlock = True
        # if no looks in previous block, append to notLookingTime
        if blockNum != 1 and blockFace[blockNum - 1] == 0:
            notLookingTime[blockNum - 1] += (list(blocks.values())[blockNum - 2][1] -
                                             list(blocks.values())[blockNum - 2][0])
        elif nl:
            notLookingTime[blockNum - 1] += list(blocks.values())[blockNum - 2][1] + diff - startnlTime
        nl = False
        notLookingTime[blockNum] = 0.0
        startnlTime = t
        if behType == "START":
            # finish look from previous block
            looks.append(look)
            lookTimes.append(videoTime)
            types.append("STOP")
            blockNumList.append("")
        behType = "POINT"
        # add baseline
        looks.append("Baseline")
        lookTimes.append(baseline + diff)
        types.append(behType)
        blockNumList.append(blockNum)
        # add new block
        look = list(blocks.values())[blockNum - 1][2]
        looks.append(look)
        lookTimes.append(round(t, 3))
        types.append(behType)
        blockNumList.append(blockNum)

    # adjusting brightness/contrast of frame
    cv.normalize(frame, frame, 0, 255, cv.NORM_MINMAX)
    if yo == "1yo":
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
    image, face = faceDetector(frame, grayFrame, lastFace, newBlock)
    frames[blockNum] += 1

    if face is not None:
        blockFace[blockNum] += 1
        # saving as most recent non-None face coordinates
        lastFace = face
        # calling landmarks detector function
        image, PointList = faceLandmarkDetector(frame, grayFrame, face)
        RightEyePoint = PointList[36:42]
        LeftEyePoint = PointList[42:48]

        # record how long child was not looking to see if that time should be considered for valid data
        # threshold not looking > 1 second; less than a second may be system error
        if nl and round(camera.get(cv.CAP_PROP_POS_MSEC) / 1000, 3) - startnlTime > 1.0:
            notLookingTime[blockNum] += round(camera.get(cv.CAP_PROP_POS_MSEC) / 1000, 3) - startnlTime
        nl = False

        # tracking eye positions across video time
        rightPos = EyeTracking(grayFrame, RightEyePoint)
        leftPos = EyeTracking(grayFrame, LeftEyePoint)
        if newBlock or right == {} or rightPos != right[videoTime] or leftPos != left[videoTime]:
            if rightPos is not None and leftPos is not None:
                # extract time of video in milliseconds; convert to seconds
                videoTime = round(camera.get(cv.CAP_PROP_POS_MSEC) / 1000, 3)
                right[videoTime] = rightPos
                left[videoTime] = leftPos
                # cases of left and right eye positions
                # opposing directions --> continue look in previous direction
                #if rightPos != "Center" and leftPos != "Center" and rightPos != leftPos:
                if rightPos != "Outside" and leftPos != "Outside" and rightPos != leftPos:
                    right[videoTime] = look
                    left[videoTime] = look
                    continue
                # both looking center/outside screen --> STOP look in previous direction
                #if rightPos == "Center" and leftPos == "Center":
                if rightPos == "Outside" and leftPos == "Outside":
                    if len(right.keys()) > 1 and blockFace[blockNum] > 1 and look != "":
                        behType = "STOP"
                    else:
                        continue
                # both one direction or one center/outside and one direction --> START look in direction
                # if STOP not written for previous look prior to start of new look, assumed to switch simultaneously
                #elif rightPos == "Center":
                elif rightPos == "Outside":
                    # exclude continued look
                    if look == "Look" + leftPos:
                        continue
                    look = "Look" + leftPos
                    behType = "START"
                else:
                    # exclude continued look
                    if look == "Look" + rightPos:
                        continue
                    look = "Look" + rightPos
                    behType = "START"
                # stop previous look direction if not already done so
                if types[-1] == "START" and behType == "START":
                    looks.append(looks[-1])
                    lookTimes.append(videoTime)
                    types.append("STOP")
                    blockNumList.append("")
                looks.append(look)
                lookTimes.append(videoTime)
                types.append(behType)
                blockNumList.append("")
                # change look to null if stopped previous
                if behType == "STOP":
                    look = ""

        # display frame
        cv.imshow('Frame', image)

    else:
        # tracking when first instance of not looking is
        if not nl:
            startnlTime = round(camera.get(cv.CAP_PROP_POS_MSEC) / 1000, 3)
            nl = True
        # display plain frame
        cv.imshow('Frame', frame)

    # adjust child coordinates if big move occurs while face isn't tracked
    cv.setMouseCallback('Frame', click_event)
    # if q is pressed on keyboard: quit
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# append final look and baseline
if behType == "START":
    looks.append(look)
    lookTimes.append(times[-1])
    types.append("STOP")
    blockNumList.append("")
looks.append("Baseline")
lookTimes.append(times[-1])
types.append("POINT")
blockNumList.append(blockNum + 1)
# if not looking, append final not looking time
if blockNum != 1 and blockFace[blockNum] == 0:
    notLookingTime[blockNum] += list(blocks.values())[blockNum - 1][1] - list(blocks.values())[blockNum - 1][0]
elif nl:
    notLookingTime[blockNum] += list(blocks.values())[blockNum - 1][1] + diff - startnlTime

# checking percentage of valid footage for each block
valid = {}
for block, nlTime in notLookingTime.items():
    valid[block] = 100 - round(nlTime / (list(blocks.values())[block - 1][1] - list(blocks.values())[block - 1][0])
                               * 100, 1)
# insert % of child looking for each block and total time for each block
#percLooking = [np.nan] * len(looks)
#timeLooking = [np.nan] * len(looks)
timeBlock = [np.nan] * len(looks)
for index, block in enumerate(blockNumList):
    if block != "":
        if looks[index] != "Baseline":
            #percLooking[index] = valid[block]
            #timeLooking[index] = (list(blocks.values())[block - 1][1] - list(blocks.values())[block - 1][0] -
            #                      notLookingTime[block])
            timeBlock[index] = list(blocks.values())[block - 1][1] - list(blocks.values())[block - 1][0]

# add positions and times to Excel file
obsID = [sys.argv[1].split("\\")[-1]] * len(looks)
# add how many trials under 60% looking to file name (i.e., how many manual coders need to code)
#lessThan60 = 0
#for perc in percLooking:
#    if perc != np.nan:
#        if float(perc) < 60:
#            lessThan60 += 1
# compute runtime of automated video coding
runtime_end = time.time()
runtime = [np.nan] * len(looks)
runtime[0] = runtime_end - runtime_start

fileName = (sys.argv[1].split("\\")[-1].split("_")[0][:3] + "_" + sys.argv[1].split("\\")[-1].split("_")[0][3:] + "_" +
            yo + "_VWM_auto_" + initials + ".xlsx")
df = pd.DataFrame({"Observation id": obsID, "Behavior": looks, "Behavior Type": types, "Modifier #1": blockNumList,
                   "Time": lookTimes, "Block Time": timeBlock,
                   "Runtime": runtime})
#"Looking Time": timeLooking, "Looking %": percLooking,
df.to_excel(fileName, sheet_name=sys.argv[1].split("\\")[-1], index=False)

# releasing video capture
camera.release()
cv.destroyAllWindows()
