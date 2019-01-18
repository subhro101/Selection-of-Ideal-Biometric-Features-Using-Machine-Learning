# File for the variance threshold feature selection algorithm
from sklearn.feature_selection import VarianceThreshold

# The function which will be called
def get_features(raw_data, raw_data_ids):
    '''
    Perform feature selection using variance threshold
    Removes features without any variance
    '''
    
    # Returns columns that meet the amount of variance
    sel = VarianceThreshold()
    sel.fit(raw_data)

     # Set aside correct columns
    return_columns = []
    index = 0
    for feature in sel.get_support():
        if feature:
            return_columns.append(index)
        index += 1

    # return
    return return_columns
