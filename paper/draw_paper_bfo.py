import matplotlib.pyplot as plt
import matplotlib.lines as lines
from pandas import read_csv
import numpy as np

## https://matplotlib.org/api/markers_api.html

## Paras

x_label = "Time (5 minutes)"
#y_label = "CPU Usage"
y_label = "Memory Usage"
#title = 'Univariate Neural Network'
title = 'Multivariate FL-BFONN'

read_filepath = "multi_ram.csv"
write_filepath = "multi_ram_flbfonn.pdf"

# CPU:  290, 750, 850, 1000, 1300  (best images)
# RAM:  290, 780, 850, 1000, 1300  (best images)
point_number = 500
point_start = 200

colnames = ['True', "FL-GANN", "FL-BFONN", "FLABL"]
results_df = read_csv(read_filepath, header=None,names=colnames, index_col=False, engine='python')

real = results_df['True'].values
flnn = results_df['FL-GANN'].values
flgann = results_df['FL-BFONN'].values
flbfonn = results_df['FLABL'].values

x = np.arange(point_number)

# plt.plot(x, real[point_start:point_start + point_number],  marker='o', label='True')
# plt.plot(x, ann[point_start:point_start + point_number],  marker='s', label='ANN')
# plt.plot(x, mlnn[point_start:point_start + point_number],  marker='*', label='MLNN')
# plt.plot(x, flnn[point_start:point_start + point_number],  marker=lines.CARETDOWN, label='FLNN')
# plt.plot(x, flgann[point_start:point_start + point_number],  marker='x', label='FL-GANN')
# plt.plot(x, flbfonn[point_start:point_start + point_number],  marker='+', label='FL-BFONN')


plt.plot(x, real[point_start:point_start + point_number], label='Actual')
#plt.plot(x, flnn[point_start:point_start + point_number],  label='Predict')
plt.plot(x, flgann[point_start:point_start + point_number], label='Predict')
#plt.plot(x, flbfonn[point_start:point_start + point_number], label='Predict')

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(title)
plt.legend()
plt.savefig(write_filepath)
plt.show()
