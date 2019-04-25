import matplotlib.pyplot as plt
import matplotlib.lines as lines
from pandas import read_csv
import numpy as np

## https://matplotlib.org/api/markers_api.html


## Paras

x_label = "Time (5 minutes)"
#y_label = "CPU Usage"
y_label = "Memory Usage"
title = 'Univariate Neural Network'
#title = 'Multivariate Neural Network'

read_filepath = "uni_ram_slid_2.csv"
write_filepath = "uni_ram_slid_2.pdf"

# CPU:  290, 750, 850, 1000, 1300  (best images)
# RAM:  290, 780, 850, 1000, 1300  (best images)
point_number = 70
point_start = 780

colnames = ['True','ANN', "MLNN", "FLNN", "FL-GANN", "FL-BFONN"]
results_df = read_csv(read_filepath, header=None,names=colnames, index_col=False, engine='python')

real = results_df['True'].values
ann = results_df['ANN'].values
mlnn = results_df['MLNN'].values
flnn = results_df['FLNN'].values
flgann = results_df['FL-GANN'].values
flbfonn = results_df['FL-BFONN'].values

x = np.arange(point_number)

plt.plot(x, real[point_start:point_start + point_number],  marker='o', label='True')
plt.plot(x, ann[point_start:point_start + point_number],  marker='s', label='ANN')
plt.plot(x, mlnn[point_start:point_start + point_number],  marker='*', label='MLNN')
plt.plot(x, flnn[point_start:point_start + point_number],  marker=lines.CARETDOWN, label='FLNN')
plt.plot(x, flgann[point_start:point_start + point_number],  marker='x', label='FL-GANN')
plt.plot(x, flbfonn[point_start:point_start + point_number],  marker='+', label='FL-BFONN')

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(title)
plt.legend()
plt.savefig(write_filepath)
plt.show()
