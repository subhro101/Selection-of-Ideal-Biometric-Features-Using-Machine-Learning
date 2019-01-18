from sklearn import svm
import numpy as np

def refine_templates(templates, inliers, features):
	# Declare a one class svm
	clf = svm.OneClassSVM()

	# Fit to the inlier
	clf.fit(inliers)

	# Check if new data is novel
	predictions = clf.predict(templates)

	# Get novel data
	refined_templates = templates[predictions == 1, :]

	# Make sure it was not all discarded
	if len(refined_templates) == 0:
		# Get all the data but with only selected columns
		refined_templates = templates

	# return
	return np.concatenate((refined_templates, inliers))