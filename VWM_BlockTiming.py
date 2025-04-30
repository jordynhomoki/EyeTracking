'''
This is a module built with the intention of tracking block timing for the LEAP project - NIRS videocoding.
Only using EDat file in combination with audio/visual of the first block's start to output timing for all blocks to
assist manual coders.
'''

import sys
import pandas as pd
from AudioDetection import blockStart, startSig
from EPrimeExtract import blocks, yo


# change start time and difference between video file and E-Prime file based on starting signal
if startSig == "1":
    start = list(blocks.keys())[0]
else:
    start = list(blocks.values())[0][0]
diff = blockStart - start

labs = []
blockNums = []
times = []
# since blocks key is not block number, need to implement a count variable to track block number
blockCount = 1
for index, value in enumerate(list(blocks.values())):
    # for each block, need baseline entry and block type entry
    labs.append("Baseline")
    labs.append(value[2])
    # 2 entries of block number
    blockNums += 2 * [blockCount]
    # baseline time either computed from block start - 8.270 or end of previous block; both need to add diff
    if blockCount == 1:
        times.append(list(blocks.keys())[index] - 8.270 + diff)
    else:
        times.append(list(blocks.values())[index - 1][1] + diff)
    # block start directly extracted from list + diff
    times.append(value[0] + diff)
    # update block count for next loop iteration
    blockCount += 1
# need to include final baseline -- end of final block
labs.append("Baseline")
blockNums.append(blockCount)
times.append(list(blocks.values())[-1][1] + diff)

fileName = (sys.argv[2].split("\\")[-1].split("_")[-1].split("-")[1][:3] + "_" +
            sys.argv[2].split("\\")[-1].split("_")[-1].split("-")[1][3:] + "_" + yo + "_VWM_BlockTimes.xlsx")
df = pd.DataFrame({"Behavior": labs, "Modifier #1": blockNums, "Time": times})
df.to_excel(fileName, sheet_name=sys.argv[1].split("\\")[-1], index=False)
