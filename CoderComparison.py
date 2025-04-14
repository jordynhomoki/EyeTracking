'''
Automized comparison between Manual VWM Coding and Python VWM Coding.
Input: Excel files associated with manual and auto VWM coding, respectively.
Output: Excel file detailing look differences between the coding styles per block (also shows comparison of trial start/
stop times).
'''

import sys
import pandas as pd
import numpy as np

def blockLooks(df):

    # automate dataframe row iteration to collect all looks
    # current block number
    block = 0
    # checking if before or after baseline
    blockCount = 0
    # temporary start and stop times and look direction
    start = 0
    stop = 0
    look = ""
    # dictionaries holding left and right looking times for a given block
    leftLook = {}
    rightLook = {}

    for index, row in df.iterrows():
        # only check rows in a given block, reset start and stop once new block hits
        if row["Modifier #1"] != "":
            num = int(row["Modifier #1"])
            if num != block:
                # check if previous block ends with a "START" look by checking if current stop time is before start time
                if stop < start:
                    if look == "LookLeft":
                        leftLook[block] += df.loc[index, "Time"] - start
                    elif look == "LookRight":
                        leftLook[block] += df.loc[index, "Time"] - start
                # reset block number and initializing manual look time
                block = num
                leftLook[block] = 0
                rightLook[block] = 0
                blockCount = 0
            else:
                blockCount = 1
                start = row["Time"]
                stop = row["Time"]
        # only permit in-block looks (not between baseline and block)
        if blockCount == 1:
            if row["Behavior Type"] == "START":
                start = row["Time"]
                look = row["Behavior"]
            elif row["Behavior Type"] == "STOP":
                stop = row["Time"]
                look = row["Behavior"]
                if look == "LookLeft":
                    leftLook[block] += stop - start
                elif look == "LookRight":
                    rightLook[block] += stop - start

    leftLook.popitem(); rightLook.popitem()
    return leftLook, rightLook


# read in manual and python Excel sheets
file = pd.ExcelFile(sys.argv[1])
manual = pd.read_excel(file).fillna("")
manual = manual.rename(columns={"Behavior type": "Behavior Type"})
file = pd.ExcelFile(sys.argv[2])
python = pd.read_excel(file).fillna("")

# dictionaries holding manual code left and right looking times for a given block
mLeft, mRight = blockLooks(manual)
# dictionaries holding python code left and right looking times for a given block
pLeft, pRight = blockLooks(python)

# extract looking % per block from python code
lookPerc = {}
block = 0
for index, row in python.iterrows():
    if row["Looking %"] != "":
        block += 1
        lookPerc[block] = row["Looking %"]

# difference in left/right looks between manual and python code blocks
# difference is with respect to manual code (i.e., manual - python)
diffLeft = {}
diffRight = {}
for block in list(mLeft.keys()):
    diffLeft[block] = mLeft[block] - pLeft[block]
    diffRight[block] = mRight[block] - pRight[block]

# write Excel output file
fileName = "_".join(sys.argv[1].split("\\")[-1].split("_")[0:4]) + "_Comparison.xlsx"
df = pd.DataFrame({"Manual L": mLeft, "Python L": pLeft, "Difference L": diffLeft,
                   "Manual R": mRight, "Python R": pRight, "Difference R": diffRight,
                   "": np.nan, "Looking % (Python)": lookPerc})
writer = pd.ExcelWriter(fileName, engine='xlsxwriter')
df.to_excel(writer, index=True, index_label="Block", sheet_name="Comparison")

# change color scheme in Excel file
workbook = writer.book
sheet = writer.sheets["Comparison"]
(max_row, max_col) = df.shape
# employ conditional formatting to highlight magnitude of differences
sheet.conditional_format(1, 3, max_row, 3, {"type": "3_color_scale",
                                            "min_type": "num", "min_value": -10, "min_color": "#F01E2C",
                                            "mid_type": "num", "mid_value": 0, "mid_color": "#FFFFFF",
                                            "max_type": "num", "max_value": 10, "max_color": "#007FFF"})
sheet.conditional_format(1, 6, max_row, 6, {"type": "3_color_scale",
                                            "min_type": "num", "min_value": -10, "min_color": "#F01E2C",
                                            "mid_type": "num", "mid_value": 0, "mid_color": "#FFFFFF",
                                            "max_type": "num", "max_value": 10, "max_color": "#007FFF"})
# color headers and block numbers
col1 = workbook.add_format({"bg_color": "#F4EBFE"})
col2 = workbook.add_format({"bg_color": "#CDA4FB"})
sheet.conditional_format(1, 0, max_row, 0, {"type": "no_blanks", "format": col1})
sheet.conditional_format(0, 0, 0, max_col, {"type": "no_blanks", "format": col2})
# adjust column width
sheet.set_column(0, 8, 16)
# add borders
border_fmt = workbook.add_format({'bottom': 2, 'top': 2, 'left': 2, 'right': 2})
sheet.conditional_format(0, 0, max_row, max_col, {'type': 'no_blanks', 'format': border_fmt})
writer.close()
