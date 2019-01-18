import numpy as np
import varience_threshold as vt
import tree_selection_features as tsf
import recursive_features as rf
import information_gain as ig
import outlier_detection_system as ods
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

def compute_dprime(genuine, imposter):
	x = np.sqrt(2) * abs(np.mean(genuine) - np.mean(imposter))
	y = np.sqrt(np.power(np.std(genuine), 2) + np.power(np.std(imposter), 2))
	return x / y
	
def optimize(templates, templates_ids):

	# log
	print("Finding optimal parameters")

	# Split the sets
	templates, validation, templates_ids, validation_ids = train_test_split(
			templates, templates_ids, test_size=0.30)
	users = np.unique(validation_ids)

	# iterate over each user
	params = []
	for user in users:

		# Declare fold variables
		selected_features = []
		performance = []

		# Remove outliers within user data
		data = validation[validation_ids == user]
		inliers = ods.remove_outliers(data)
		ids = (validation_ids == user) * 1

		# Get features
		temp = vt.get_features(validation, ids)
		selected_features.extend(temp)
		temp = tsf.get_features(validation, ids)
		selected_features.extend(temp)
		temp = rf.get_features(validation, ids)
		selected_features.extend(temp)
		temp = ig.get_features(validation, ids)
		selected_features.extend(temp)

		# Sort features by number of times selected
		feature_counts = np.zeros(len(validation))
		for selection in selected_features:
			feature_counts[selection] += 1
		sorted_index = np.argsort(-1 * feature_counts)

		# For every feature
		for k in range(1, len(sorted_index)):
			# Declare feature scope variables
			gen_scores = []
			imp_scores = []

			# If this feature was never selected, break
			if feature_counts[sorted_index[k]] == 0:
				break

			# For each user
			for this_user in range(len(users)):
				# For each user
				for other_user in range(len(users)):
					# if this user is the current iteration
					if users[this_user] == user:
						# Calculate distance
						dist = distance.chebyshev(validation[this_user, sorted_index[0:k]],
							validation[other_user, sorted_index[0:k]])

						# other user is also the current iteration
						if users[other_user] == user:
							gen_scores.append(dist)
						else:
							imp_scores.append(dist)

			# Compute d-prime
			dp = compute_dprime(gen_scores, imp_scores)
			performance.append(dp)

		# Obtain the best subset of features
		best_k = np.argmax(performance)
		if best_k == 0: best_k = 1
		feats = sorted_index[0:best_k]
		params.append((user, feats, inliers))

	# Log
	print("Optimization finished")

	# Return
	return params, templates, templates_ids
