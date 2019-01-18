# File for the Tree Based feature selection algorithm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import numpy as np

# The function that will be called
def get_features(raw_data, raw_data_ids, debug=0):
    '''
    Uses tree selection to determine feature importance
    then used the avg of the importance to remove unneeded features
    '''

    # Create a tree classifier
    clf = ExtraTreesClassifier(n_estimators=100)
    clf.fit(raw_data, raw_data_ids)

    # Calculate feature importance
    model = SelectFromModel(clf, prefit=True)

    # Set aside correct columns
    return_columns = []
    index = 0
    for feature in model.get_support():
    	if feature:
    		return_columns.append(index)
    	index += 1

    # DEBUG
    if debug == 1 :
    	importances = clf.feature_importances_
    	std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    	indices = np.argsort(importances)[::-1]
    	plt.figure(figsize=(16,8))
    	plt.title("Feature importances")
    	plt.bar(range(raw_data.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    	plt.xticks(range(raw_data.shape[1]), indices)
    	plt.xlim([-1, raw_data.shape[1]])
    	plt.savefig('./RESULTS/IMAGES/tree_feature_importance.png', bbox_inches='tight')
    	plt.show()

    # return
    return return_columns
