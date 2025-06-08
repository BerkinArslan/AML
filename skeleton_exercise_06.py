"""
Skeleton for exercise 06: implementing a full decision tree from scratch

DISCLAIMER:
Please note that this solution may contain errors (please report them, thanks!), and that there are most-likely more
elegant and more efficient implementations available for the given problem. In this light, this solution may only serve
as an inspiration for solving the given exercise tasks.

(C) Merten Stender, TU Berlin
merten.stender@tu-berlin.de
"""
import numpy as np
from matplotlib import pyplot as plt


class Node:
    """Node class for the decision tree.
    A node can be a leaf node (with a prediction value) or a parent node having
    two child nodes as attributes (with a splitting condition and child nodes).
    """
   
    def __init__(self,
                 split_dim: int = None, 
                 split_val: float = None,
                 child_left=None, child_right=None,
                 prediction: int = None):
        """Initialize Node class
        If a split condition and child nodes are provided: returns a parent
        node for further splitting. 
        Otherwise: returns a leaf node making the <prediction>
        """

        # splitting condition
        self.split_dim = split_dim  # the index of the feature dimension
        self.split_val = split_val  # the value along which to split x <= value

        # child nodes to the current node
        self.child_left = child_left    # the left child node
        self.child_right = child_right  # the right child node

        # prediction (if leaf) value
        # the output value \hat{y} of current node
        self.prediction = prediction

    def is_leaf(self):
        """ returns boolean to indicate leaf node"""
        return self.prediction is not None


"""
Some static helper functions
"""


def majority_vote(y: np.ndarray) -> int:
    """ Return majority vote
    for categorical data in y, i.e. the index of the most common class
    """

    # TASK 2.1
    unique_labels = np.unique(y)
    n_unique = unique_labels.shape[0]
    max = 0
    max_index = 0
    for i in range(n_unique):
        n = y[y == unique_labels[i]].shape[0]
        if max < n:
            max = n
            max_index = i
    majority_class = unique_labels[max_index]
    return majority_class


def _entropy(y: np.ndarray) -> float:
    """ Compute Shannon information entropy"""
    proportions = np.bincount(y.astype(dtype=int)) / len(y)
    return -np.sum([p * np.log2(p) for p in proportions if p > 0])


def _information_gain(y_parent: np.ndarray, index_split: np.ndarray) -> float:
    """Compute information gain for given split of categorical data"""

    # number of members per child node
    n = len(index_split)  # overall number of data points
    n_left = np.sum(index_split == 1)  # members of left child
    n_right = np.sum(index_split == 0)  # members of right child

    # compute entropy at parent node
    H_parent = _entropy(y_parent)

    # information gain will be zero if a child has no members (special case)
    if n_left == 0 or n_right == 0:
        return 0

    # compute information gain
    H_left = _entropy(y_parent[index_split])
    H_right = _entropy(y_parent[index_split == 0])

    info_gain = H_parent - ((n_left / n) * H_left + (n_right / n) * H_right)

    # print(f'H-parent = {H_parent}, H_left={H_left:.3f}, H_right={H_right:.3f}, info gain={info_gain:.3f}')
    return info_gain


def _create_split(x: np.ndarray, split_dim: int, split_val: float) -> np.ndarray:
    """Split data set X according to split condition"""
    # returns an index where x<= split value is true (left child node)
    # expects X to be of shape [nbatch, n_feat_dims]
    return x[:, split_dim] <= split_val


"""
The DecisionTree class
"""


def _best_split(x: np.ndarray, y: np.ndarray):
    """Find the best split w.r.t. information gain"""

    split = {'score': 0, 'dim': None, 'thresh': None}
    num_feats = x.shape[1]
    for _split_dim in range(num_feats):  # loop through feature space dimensions
        possible_splits = np.unique(x[:, _split_dim])  # all possible splits along this feature dimension
        for _split_val in possible_splits:  # loop through possible splits in current feat. dimension

            # create split and compute information gain
            idx_split = _create_split(x=x, split_dim=_split_dim, split_val=_split_val)
            score = _information_gain(y, idx_split)

            # update if score was better than before
            if score > split['score']:
                split['score'] = score
                split['dim'] = _split_dim
                split['thresh'] = _split_val

    print(
        f'best split: feat. dim={split["dim"]}, feat. value={split["thresh"]:.3f}, info. gain={split["score"]:.3f}')

    return split['dim'], split['thresh']


class DecisionTree:

    def __init__(self, max_depth: int = 5, min_samples: int = 2):
        """Initialize DecisionTree class"""

        self.root = None  # store the root node here
        self.max_depth = max_depth  # maximum depth of the tree
        self.min_samples = min_samples  # minimum number of samples per node

        self.class_labels = None  # list of class labels
        self.n_samples = None  # number of training data samples
        self.n_features = None  # feature space dimensionality

        #  <-- this is where we would also put choices about the splitting
        # condition, i.e. Gini or entropy / log


        # Attributes used during tree growth
        self._curr_no_samples: int = None  # number of samples in current node
        self.is_completed: bool = None  # stopping criteria for tree growth

    def terminate(self, depth):
        """ Check tree growth stopping criteria"""

        # return True if one of the following conditions is true
        # - max depth reached
        # - minimal number of samples in a node
        # - node is pure

        if (depth >= self.max_depth) \
            or (self._curr_no_samples < self.min_samples) \
                or self._curr_node_pure:
            return True

        return False

    def _grow_tree(self, x: np.ndarray, y: np.ndarray, curr_depth: int = 0):
        """ Build tree for given training data set
        This is a recursive function that calls itself over and over until it
        hits the stopping criteria. This method creates nodes and child nodes.

        Recursive function: traverses input and target data through a tree, 
        thereby creating nodes and the tree itself. Returns a node for the 
        given input data set.
    
        x: [N*, n*] data set of N* samples with n* feature dimensions
        y: [N*] labels
    
        Note that (*) the number of samples differs from node to node, hence
        N* < N is the general case, where N is the overall number of samples
        in the training data set.
        """
    
        # current number of samples (for N_min stopping condition in is_completed())
        self._curr_no_samples = x.shape[0]

        # check for purity in the current set of labels y. True if pure -> stop growth
        self._curr_node_pure = len(np.unique(y)) == 1

        # TASK 2.2
        # base condition (stopping criteria) for recursive call.
        if self.terminate(curr_depth):
            # return a leaf node carrying the majority vote if base condition is met
            return Node(prediction = majority_vote(y))

        # if base condition is not met:
        # 1. recursively grow the tree by finding the optimal split for the current input and output data
        split_dim, split_val = _best_split(x, y)
        print(f'depth={curr_depth}; split condition: x_{split_dim} <= {split_val:.3f}\n')

        # 2.split the current data according to best split: obtain split index
        left_idx =_create_split(x, split_dim, split_val)  # data for left child node
        right_idx = ~ left_idx                              # data for right child node

        # 3. create left+right child nodes and assign data. Make recursive call on each child node
        child_left = self._grow_tree(x[left_idx, :], y[left_idx], curr_depth = curr_depth + 1)
        child_right = self._grow_tree(x[right_idx, :], y[right_idx], curr_depth=curr_depth + 1)

        # create the current node (with current split condition). Link to child nodes created in previous line
        node = Node(split_dim=split_dim, split_val=split_val,
                    child_left=child_left, child_right=child_right)

        return node

    def _traverse_tree(self, x: np.ndarray, node: Node = None) -> int:
        """ Feed data trough given tree structure.
        This is a recursive function that feeds an incoming data sample x
        through the tree until a leaf node is met, for which the prediction
        value is returned. This function navigates the tree

        x: [1, n] individual data sample
        returns scalar, i.e. predicted class label
        """

        # base condition for breaking the recursive call
        if node.is_leaf():
        # TASK 2.3
            return node.prediction # if leaf node: return majority vote prediction

        # if node is not leaf, further traverse the tree (RECURSIVE FUNCTION)
        #x is an individual data point not a data set
        if x[node.split_dim] <= node.split_val:
            return self._traverse_tree(x, node.child_left)
        else:
            return self._traverse_tree(x, node.child_right)

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit tree to training data.
        Generate root node and then build the tree until hitting the stopping
        criterion or leaf nodes.

        x: [N, n] data set of N samples with n feature dimensions
        y: [N] labels for the N samples
        """

        # data properties: size, dimensionality, categories, list of all class labels
        self.n_samples, self.n_features = x.shape
        self.class_labels = np.unique(y)

        # grow the tree and append nodes to root node
        # TASK 2.4
        #why are we putting the whole tree in the root and not just one root?
        #because it is returning a node and recursively making other nodes
        #but the first returned node is the root
        self.root = self._grow_tree(x, y, curr_depth=0)

    def predict(self, x: np.ndarray):
        """Make a prediction for unseen data.

        Traverse the tree with some new data and return the prediction.

        x: [N, n] data set of N samples with n feature dimensions
        returns: [N] predicted class labels for the N samples
        """
        # feed each data sample in x through the tree
        # TASK 2.5
        predictions = [self._traverse_tree(data_point, self.root) for data_point in x]

        return np.array(predictions)


if __name__ == '__main__':

    """
    Simple 1-dimensional data set for de-bugging your code. 
    We should split at -0.5, and use a DT of depth=1
    """
    # x_train = np.expand_dims(np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]), axis=-1)
    # y_train = np.array([0, 0, 0, 1, 1, 1])

    # # validation data
    # x_val = np.expand_dims(np.array([-1.5, 1.8]), axis=-1)
    # y_val = np.array([0, 1])

    """
    Exercise data set (same as in DBSCAN lecture)
    (everything within Manhatten-Norm<=1 should be class 1)
    """
    # data set for the exercise (same as displayed in lecture on DBSCAN)
    data = np.loadtxt('decision_tree_dataset.txt', delimiter=',')
    x_train = data[:, :2]               # features
    y_train = data[:, -1].astype(int)   # targets

    """
    Fit decision tree to training data 

    Feel free to experiment with the hyperparameters max_depth and min_samples!
    """
    # create DT class object
    DT = DecisionTree()  # potentially constrain depth or min_samples

    # fit DT to data set
    DT.fit(x=x_train, y=y_train)

    """
    Investigate the DT after training:
        - how well does it perform on the training data set
        - how well does it generalize (new, unseen data)
        - illustrate the decision boundaries
    """
    # make prediction (first on training data set)
    x_val = x_train
    y_val = y_train
    y_pred = DT.predict(x_val)

    print(f'\n\nground truth labels: \t {y_val}')
    print(f'predicted labels: \t \t {y_pred}')

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(x_val[y_val == 0, 0], x_val[y_val == 0, 1], linestyle='none', marker='.', markersize=10, )  # color='blue')
    plt.plot(x_val[y_val == 1, 0], x_val[y_val == 1, 1], linestyle='none', marker='*', markersize=10, )  # color='red')
    plt.legend(['class 0', 'class 1'])
    plt.title('ground truth')
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')

    plt.subplot(1, 2, 2)
    plt.plot(x_val[y_pred == 0, 0], x_val[y_pred == 0, 1], linestyle='none', marker='o',
             markersize=5, )  # color='blue')
    plt.plot(x_val[y_pred == 1, 0], x_val[y_pred == 1, 1], linestyle='none', marker='+',
             markersize=15, )  # color='red')
    plt.legend(['class 0', 'class 1'])
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.title('prediction')
    plt.tight_layout()
    plt.show()

    # generate some unseen data
    # (everything within Manhatten-Norm<=1 should be class 1)
    x_val2 = np.array([[0, 0.8], [0, 1.2], [2, 0]])
    y_val2 = np.array([1, 0, 0])
    y_pred2 = DT.predict(x_val2)

    x_val = np.concatenate((x_val, x_val2))
    y_val = np.concatenate((y_val, y_val2))
    y_pred = np.concatenate((y_pred, y_pred2))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(x_val[y_val == 0, 0], x_val[y_val == 0, 1], linestyle='none', marker='.', markersize=10)
    plt.plot(x_val[y_val == 1, 0], x_val[y_val == 1, 1], linestyle='none', marker='*', markersize=10)
    plt.legend(['class 0', 'class 1'])
    plt.title('new data: ground truth')
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')

    plt.subplot(1, 2, 2)
    plt.plot(x_val[y_pred == 0, 0], x_val[y_pred == 0, 1], linestyle='none', marker='o', markersize=10)
    plt.plot(x_val[y_pred == 1, 0], x_val[y_pred == 1, 1], linestyle='none', marker='+', markersize=10)
    plt.legend(['class 0', 'class 1'])
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.title('new data: prediction')
    plt.tight_layout()
    plt.show()

    """
    Visualize the decision boundaries by quering the DT for a grid of points
    """
    # generate some grid-like data to visualize the decision boundaries learned
    x1_grid = np.arange(-3, 3, 0.1)
    x2_grid = x1_grid
    x_grid = []
    for x1 in x1_grid:
        for x2 in x2_grid:
            x_grid.append([x1, x2])
    x_grid = np.array(x_grid)

    # make predictions for all grid data points
    y_pred_grid = DT.predict(x_grid)

    fig = plt.figure()
    plt.plot(x_grid[y_pred_grid == 0, 0], x_grid[y_pred_grid == 0, 1], linestyle='none', marker='.', markersize=10,
             alpha=0.5)
    plt.plot(x_grid[y_pred_grid == 1, 0], x_grid[y_pred_grid == 1, 1], linestyle='none', marker='.', markersize=10,
             alpha=0.5)
    plt.legend(['class 0', 'class 1'])
    plt.title('decision boundaries learned by DT')
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.show()
