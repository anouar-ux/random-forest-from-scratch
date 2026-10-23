"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    if len(labels)<2:
        return 0

    L,C=np.unique(labels, return_counts=True)
    C=np.array(C)/len(labels)

    S=float(np.sum(C**2))

    return 1-S

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    col = features[:,feature_index]
    mask = col <=threshold
    L= features[mask]
    y_L= labels[mask]
    R= features[~mask]
    y_R=labels[~mask]

    return (L,y_L,R,y_R)

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.

    if len(parent_labels)==0:
        return None


    P=len(parent_labels)
    L=len(left_labels)
    R=len(right_labels)

    P_imp= impurity(parent_labels)
    L_imp=impurity(left_labels)
    R_imp=impurity(right_labels)

    gain= P_imp-(L*L_imp/P + R*R_imp/P)

    return gain

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.
    
    result={'feature_index': None, 'threshold': None, 'score': 0.0}

    for k in feature_indices:
        col=features[:,k]
        sorted_col=np.sort(np.unique(col))
        N=len(sorted_col)
        if N==1:
            continue
        for l in range(N-1):
            threshold=(sorted_col[l]+sorted_col[l+1])/2
            (L,y_L,R,y_R) =split_dataset(features, labels, k, threshold)
            score = split_score(labels,y_L,y_R)
            if score>result['score']:
                result={'feature_index': k, 'threshold': threshold, 'score': score}
    
    
    return result

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    return depth==max_depth or len(labels)<min_samples_split or len(np.unique(labels))==1

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    uniques,counts=np.unique(labels, return_counts=True)
    return int(uniques[np.argmax(counts)])

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    # TODO: recursively grow a decision tree, returning a nested dict of leaf/internal nodes.
    dico={}

    
    if should_stop(labels,depth,max_depth,min_samples_split):
        dico['leaf']=True
        dico['prediction']=leaf_prediction(labels)
        return dico
    
    
    if feature_subset is None:
        feature_subset=range(features.shape[1])
    result=best_split(features, labels, feature_subset)
    
    (L,L_labels,R,R_labels)=split_dataset(features, labels, result['feature_index'], result['threshold'])

    dico['leaf']=False
    dico['feature_index']=result['feature_index']
    dico['threshold']=result['threshold']
    dico['left']=build_tree(L, L_labels, max_depth, min_samples_split, feature_subset, depth+1)
    dico['right']=build_tree(R, R_labels, max_depth, min_samples_split, feature_subset, depth+1)

    return dico

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # TODO: walk the example down the fitted tree until you reach a leaf, then return its prediction.
    if tree['leaf']:
        return tree['prediction']

    value=example[tree['feature_index']]
    if value<=tree['threshold']:
        return predict_example_tree(tree['left'],example)

    return predict_example_tree(tree['right'],example)

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    # TODO: return predicted class for each row of features using the fitted tree.
    n=features.shape[0]
    result=np.zeros(n)
    for k in range(n):
        result[k]= predict_example_tree(tree, features[k])

    return result

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    # TODO: draw a bootstrap sample of rows (with replacement) using rng.
    n=features.shape[0]

    idx=rng.integers(0, n, size=n)

    return (features[idx],labels[idx])

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    # TODO: return num_to_pick distinct random feature indices from range(num_features) using rng.
    idx=rng.permutation(num_features)[:num_to_pick]
    
    return idx

# Step 12 - train_forest
import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    # TODO: grow num_trees decision trees on bootstrap samples with random feature subsets.

    n,m=features.shape
    rng=np.random.default_rng(random_state)
    result=[]

    if num_features_per_split is None:
        num_features_per_split=max(1,round(m**0.5))


    for _ in range(num_trees):
        X_boot,y_boot=bootstrap_sample(features, labels, rng)
        features_sub=feature_subset(m,num_features_per_split,rng)
        tree=build_tree(X_boot,y_boot,max_depth,min_samples_split,features_sub,depth=0)
        dico={'tree':tree, 'feature_indices':features_sub}
        result.append(dico)

    return result

# Step 13 - combine_predictions
def combine_predictions(tree_predictions):
    # TODO: aggregate the per-tree predictions of an ensemble into one prediction per example.
    
    t,n=tree_predictions.shape

    result=np.zeros(n)

    for k in range(n):
        col=tree_predictions[:,k]
        unique,count=np.unique(col, return_counts=True)
        result[k]=unique[np.argmax(count)]

    return result

# Step 14 - predict_forest
def predict_forest(forest, features):
    # TODO: predict classes for a dataset using the whole trained forest.

    T,n=len(forest),features.shape[0]
    tree_predictions=np.zeros((T,n))

    result=np.zeros(n)

    for k in range(len(forest)):
        dico=forest[k]
        tree=dico['tree']
        tree_predictions[k]=predict_tree(tree,features)

    return combine_predictions(tree_predictions)

# Step 15 - accuracy (not yet solved)
# TODO: implement

