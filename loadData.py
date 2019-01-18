from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pandas as pd

def loadData(path):
	# log
	print('Loading data')

	# declare variables
	raw_data = []
	raw_data_ids = []
	ids = 0
	temp_list = []

	# iterate over each users file
	files = os.listdir(path)
	for f in files:
		temp = pd.read_csv(path + f, header=None)
		temp_list.append(temp)
		raw_data_ids.extend([ids] * len(temp.index))
		ids += 1

	# flatten the lists
	raw_data = np.vstack(temp_list)
	raw_data = raw_data.astype(np.float64)
	raw_data_ids = np.array(raw_data_ids)

	# Set aside unique ids and scale data
	unique_ids = np.unique(raw_data_ids)
	scaler = MinMaxScaler()
	raw_data = scaler.fit_transform(raw_data)

	# log
	print("Raw Rows: ", len(raw_data))
	print("Users: ", len(files))
	print("Data loaded and scaled")

	# return
	return raw_data, raw_data_ids, unique_ids