'''
Extracting information from the E-Prime file for use in VWM_EyeTracking.py, HND_EyeTracking.py.
'''

import sys

# change path depending on subject number/age
# need to add second "\" whenever there is a "\" in file name
path = sys.argv[-1]
file = open(path, "r", encoding="UTF-16")
file = file.read().split("\n")
# identify if coding is for VWM or HND trials
trialType = path.split("\\")[-1][:3]

# lists of block features and dictionary to divide trials
trials = []
blockTypes = []
blocks = {}

# block identification for VWM
if trialType == "VWM":
    for line in file:
        if "\t\tStim.OnsetTime" in line:
            # time of block stim
            trials.append(int(line[18: len(line)]) / 1000)
        elif "\tDirection" in line:
            if not trials:
                continue
            # box changes in left or right side
            blockTypes.append([line[12]])
        elif "\tLoadNumber" in line:
            if not trials:
                continue
            # number/load of boxes that change
            load = "Low"
            # low/med/high differs between 1yo and 2yo/3yo
            yo = sys.argv[1].split("\\")[-1].split("_")[1]
            if yo == "12mo":
                yo = "1yo"
                if line[13] == "2":
                    load = "Med"
                elif line[13] == "3":
                    load = "High"
            else:
                # in case videos are tagged incorrectly (have seen examples of this)
                if yo == "24mo":
                    yo = "2yo"
                elif yo == "36mo":
                    yo = "3yo"
                if line[13] == "4":
                    load = "Med"
                elif line[13] == "6":
                    load = "High"
            blockTypes[-1].append(load)
        elif "\tAttention.OnsetTime" in line:
            # no blocks shown for a given attention start
            if not trials:
                continue
            # append attention start with block stim start and blank end times
            blocks[round(int(line[22: len(line)]) / 1000, 3)] = [trials[0], trials[len(trials) - 1] + .250,
                                                                 blockTypes[-1][1] + blockTypes[-1][0]]
            trials = []

# block identification for HND
elif trialType == "HND":
    for line in file:
        if "\t\tTrialList:" in line:
            # numbered trial
            trials.append(int(line[13: len(line)]))
            blocks[trials[-1]] = [[], ""]
        elif "\t\tSoundOFF.OnsetTime" in line:
            # time trial begins
            blocks[trials[-1]][0].append(int(line[22: len(line)]) / 1000)
        elif "\t\tSoundON.OnsetTime" in line:
            # time first sentence actually starts
            blocks[trials[-1]][0].append(int(line[21: len(line)]) / 1000)
        elif "\t\tSoundON.OffsetTime" in line:
            # time trial ends
            blocks[trials[-1]][0].append(int(line[22: len(line)]) / 1000)
        elif "\t\tNIRSTag" in line:
            # trial type
            typ = "Familiar"
            if line[11] == "N":
                typ = "Novel"
            elif line[11] == "R":
                typ = "Repeat"
            blocks[trials[-1]][1] = typ

for i, time in enumerate(blocks.keys()):
    for nextTime in list(blocks.keys())[i + 1: i + 2]:
        if nextTime - time > 30:
            if trialType == "VWM":
                if (i + 1) % 4 == 0:
                    continue
                print("\nWARNING: Error in trial timing (between blocks {0} and {1}). Check NIRS spreadsheet for trial "
                      "abnormalities.".format(i + 1, i + 2))
            elif trialType == "HND":
                print("\nWARNING: Error in trial timing (between trials {0} and {1}). Check NIRS spreadsheet for trial "
                      "abnormalities.".format(i + 1, i + 2))
            q = input("Type \"Y\" and press Enter to quit session (otherwise type anything and press Enter). ")
            if q == "Y":
                quit()
