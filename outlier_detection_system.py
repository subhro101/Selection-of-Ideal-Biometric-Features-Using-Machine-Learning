## Outliers detection implemented with Isolation Forest

# Import required Libraries
import numpy as np
from sklearn.ensemble import IsolationForest

def remove_outliers(template):

    # Set variables
    inliner = []
    clf = IsolationForest(contamination=0.0)
    count = 0

    for i in range(template.shape[0]):
        # remove the query in question, returns the array without the query
        temp = np.delete(template, i, axis=0)

        # train on the templates without the query
        clf.fit(temp)

        # predict if the query in question is novel
        prediction = clf.predict(template[i, :].reshape(1, -1))

        # check the prediction
        if prediction == 1:
            inliner.append(template[i, :])
            count += 1

    # Check if all rows where outliers
    if len(inliner) > 0:
        inliner = np.stack(inliner)
    else:
        inliner = template

    # return
    return inliner
