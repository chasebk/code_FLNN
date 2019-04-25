import pandas as pd
import numpy as np
from utils import IOUtil
from utils.IOUtil import save_preprocessing_data

# Get Date: https://www.epochconverter.com/
# Get Duration: https://www.timeanddate.com/date/durationresult.html

gg = "data_resource_usage_5Minutes_6176858948_formatted.csv"
gg_format = "./"

cols = [3, 4]

df = pd.read_csv(gg, usecols=cols, header=None, index_col=False)
dataset_original = np.array(df.values)
num_rows = len(dataset_original)

min1, min2 = np.where(dataset_original[:,0:1] == 0), np.where(dataset_original[:,1:2] == 0)

### Replace value -1 = Mean
for i in range(1, len(cols)):
    count = np.count_nonzero(dataset_original[:,i:i+1] == 0)
    value_change = (np.sum(dataset_original[:,i:i+1], axis=0) + count) / num_rows
    dataset_original[:,i:i+1] = np.where(dataset_original[:,i:i+1] == -1, value_change, dataset_original[:,i:i+1])

### Calculate StartTime and EndTime
for row in dataset_original:
    row[1] = row[0] + row[1]
    row[2] = row[1] + row[2]

dataset_original = dataset_original[:,1:]       # Remove initial values (zeros)


### Sort by Time
dataset_original = np.array(sorted(dataset_original, key=lambda x: x[0]))

IOUtil.save_preprocessing_data(dataset_original, pathfile=gg_format)
print("Preprocessing Data Done")