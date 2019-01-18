# Made by: Carlos Leon, Subhrajyoti Pradhan
# Strong Thanks to Dr. Neil Tempest

## Import required libraries
import numpy as np
import warnings
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

## Import data management
import loadData as ld
import performance as p
import novelty_detection_system as nd
from templateOptimization import optimize

## Ignore Warnings
warnings.filterwarnings("ignore")

## Declare Variables
dataset_path = "dataset/data/"
k_folds = 5
counter = 1
fold_performance = []

## Load the dataset
raw_data, raw_data_ids, unique_ids = ld.loadData(dataset_path)

## Determine parameters
# Determine range
range_of_k = []
middle = int(np.floor(np.sqrt(len(unique_ids))))
if middle % 2 == 0:
    middle += 1
for i in range(middle - 5, middle + 5):
    if i > 0 and i % 2 != 0:
        range_of_k.append(i)

# Instantiate classifiers
parameters = {"n_neighbors": range_of_k}
knn = KNeighborsClassifier(n_neighbors=3)

## Perform k-fold cross validation
kf = KFold(n_splits=k_folds, shuffle=True)
for train, test in kf.split(raw_data, raw_data_ids):

    # log
    print("Fold ", counter, " of ", k_folds)
    counter += 1

    # Create fold variables 
    genuine_scores = []
    imposter_scores = []

    # Separate fold data
    templates = raw_data[train, :]
    template_ids = raw_data_ids[train]
    queries = raw_data[test, :]
    query_ids = raw_data_ids[test]

    # Retrieve user specific ideal parameters
    params, templates, template_ids = optimize(templates, template_ids)

    # Generate training data per user
    training_data = []
    training_ids = []
    for uid in unique_ids:
        # Get optimal parameters
        param = [p for p in params if p[0] == uid]
        features = param[0][1]
        inliers = param[0][2]

        # Get template
        t = templates[template_ids == uid]
        t = nd.refine_templates(t, inliers, features)

        # Set aside the data
        training_data.extend(t)
        training_ids.extend([param[0][0]] * len(t))

    # Stack data together
    training_data = np.array(training_data)

    # log
    print("Starting training/prediction")

    for uid in unique_ids:
        # Get optimal parameters for relevant columns
        param = [p for p in params if p[0] == uid]
        q = queries[:, param[0][1]]
        t = training_data[:, param[0][1]]

        # Fit to KNN using relevant columns
        clf = GridSearchCV(knn, parameters, cv=k_folds)
        clf.fit(t, training_ids)

        # predict
        predictions = clf.predict(q)
        confidence = clf.predict_proba(q)

        # Save predictions
        for i in range(len(predictions)):
            # Append the scores
            if predictions[i] == query_ids[i]:
                genuine_scores.append(confidence[i][predictions[i]])
            else:
                imposter_scores.append(confidence[i][predictions[i]])

    # log
    print("Finished training/prediction")

    # Measure fold performance
    import pdb; pdb.set_trace()  # breakpoint 95dbf694 //
    all_scores = genuine_scores + imposter_scores
    eer, far, frr, tpr = p.getScores(genuine_scores, imposter_scores)
    fold_performance.append((eer, far, frr, tpr))

    # Plot
    p.plot_scoreDist(genuine_scores, imposter_scores, name=counter)


# Plot Performance
p.plot_det(fold_performance, name="knn-chebyshev")
p.plot_roc(fold_performance, name="knn-chebyshev")
