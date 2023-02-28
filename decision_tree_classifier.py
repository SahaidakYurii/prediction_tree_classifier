"""
Implement a decision tree classifier
X is a np.array of feature variables
y is a np.arrat of classes of elements
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Node:

    def __init__(self, X, y, gini):
        self.X = X
        self.y = y
        self.gini = gini
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.predicted_class = None

class MyDecisionTreeClassifier:
    def __init__(self, max_depth):
        self.max_depth = max_depth
    
    def gini(self, classes):
        '''
        A Gini score gives an idea of how good a split is by how mixed the
        classes are in the two groups created by the split.
        
        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50
        classes in each group result in a Gini score of 0.5
        (for a 2 class problem).

        >>> import numpy as np
        >>> x = np.array([0,0,0,0])
        >>> a = MyDecisionTreeClassifier(10)
        >>> a.gini(x)
        0.0

        >>> import numpy as np
        >>> x = np.array([0,1,1,0])
        >>> a = MyDecisionTreeClassifier(10)
        >>> a.gini(x)
        0.5

        >>> import numpy as np
        >>> x = np.array([0,1,1,1])
        >>> a = MyDecisionTreeClassifier(10)
        >>> a.gini(x)
        0.375
        '''
        # Gets amount of elements in X database
        num = len(classes)

        # Gets amount of different classes in y database.
        # As they are represented as numbers from 0 to n, the amount is n + 
        classes_num = max(classes) + 1

        # Counts number of elements of each class and saves it to list
        cl_count = [0 for _ in range(classes_num)]
        for cl in list(classes):
            cl_count[cl] += 1

        # # Counts number of elements of each class and saves it to list
        # cl_count = list(np.unique(classes, return_counts=True)[1])

        return 1 - sum((counter / num) ** 2 for counter in cl_count)
    
    def split_data(self, X, y) -> tuple[int, int]:
        """
        Generates all possible divisions and counts their gini.
        Returns features index and threshold value that leads to a division with the smallest gini

        If no division lead to smalled gini than the base one, returns None, None
        
        >>> import numpy as np
        >>> X = np.array([(0.1, 0.2), (0.5, 0.2), (0.6, 0.3), (0.2, 0.3)])
        >>> y = np.array([0, 1, 1, 0])
        >>> a = MyDecisionTreeClassifier(10)
        >>> a.split_data(X, y)
        (0, 0.35)

        >>> import numpy as np
        >>> X = np.array([(0.1, 0.2), (0.5, 0.2), (0.6, 0.3), (0.2, 0.3)])
        >>> y = np.array([0, 0, 0, 0])
        >>> a = MyDecisionTreeClassifier(10)
        >>> a.split_data(X, y)
        (None, None)
        """
        classes_num = max(y) + 1
        features_num = len(X[0])
        elements_num = len(X)

        # Gets number of elements of each feature
        base_cl =  [0 for _ in range(classes_num)]
        for cl in list(y):
            base_cl[cl] += 1
    
        base_gini = self.gini(y)

        feature_idx = None
        threshold = None

        for feature in range(features_num):
            # gets feature values and class type of each element sorted by the value 
            values, classes = zip(*sorted(zip(X[:, feature], y)))

            # number of elements of each feature in left and right nodes
            left_cl = [0 for _ in range(classes_num)]
            right_cl = base_cl[:]

            # iterates through all possible devisions of current sorted by feater list
            for i, cl in enumerate(classes[:-1]):
                # as the iteration moves by only one element it is enough to find out class of this
                # element and change its position from right_cl to left_cl
                left_cl[cl] += 1
                right_cl[cl] -= 1

                left_gini = 1 - sum((counter / (i + 1)) ** 2 for counter in left_cl)
                right_gini = 1 - sum((counter / (elements_num - i - 1)) ** 2 for counter in right_cl)
                
                gini = (left_gini * (i + 1) + right_gini * (elements_num - i - 1)) / elements_num

                # if current division has better(smaller) gini then previous one, function saves it's result
                if gini < base_gini:
                    base_gini = gini

                    # threshold is average feature value of elements by which devision is made
                    threshold = (values[i] + values[i + 1]) / 2
                    feature_idx = feature

        return (feature_idx, threshold)
    
    def build_tree(self, X, y, depth = 0):
        """
        Recursively builds tree committing best possible devisions till
        recursive depth is not reached or no better division is possible
        """
        # generates new node
        curent_node = Node(X, y, self.gini(y))

        classes_num = max(y) + 1  
        base_cl =  [0 for _ in range(classes_num)]
        for cl in list(y):
            base_cl[cl] += 1

        # pred_cl is class, that is represented by the largest amount of elements
        # for [0,0,1,2,0] devision pred_cl is 0
        pred_cl = np.argmax(base_cl)

        if depth < self.max_depth:
            feature_idx, threshold = self.split_data(X, y)

            if feature_idx is not None:
                # if it is possible the function creates new node
                curent_node.feature_index = feature_idx
                curent_node.threshold = threshold

                # gets list where values are marked True if they are smaller that threshold
                # and False if bigger
                left_idxes = X[:, feature_idx] <= threshold
                
                # checks if both nodes have elements
                # if the tree gets X = [0.1, 0.2, 0.2] and y = [0, 0, 1]
                # threshold will be set to 0.2 and in this situation 
                # left node will get all elements when right none
                if left_idxes.any() and (~left_idxes).any():
                    curent_node.left = self.build_tree(X[left_idxes], y[left_idxes], depth + 1)
                    curent_node.right = self.build_tree(X[~left_idxes], y[~left_idxes], depth + 1)
                    return curent_node

        # saves predicted_class only for leafs of the tree
        curent_node.predicted_class = pred_cl
        return curent_node
    
    def fit(self, X, y):
        """
        Wrapper for build_tree
        """
        self.tree = self.build_tree(X, y)
    
    def predict(self, X_test):
        """
        Uses already generated decision tree saved as Node element
        to predict classes of new elements  
        """
        result = []
        # iterates through values and predicts their classes
        for vals in X_test:
            # gets decision tree
            node = self.tree

            # while it is possible goes down by the tree
            while node.left:
                if vals[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right

            # when it reaches leafs predicts elements class as leafs class 
            result.append(node.predicted_class)

        return np.array(result)
        
    def evaluate(self, X_test, y_test):
        """
        Generates result of prediction and compares it to classes of the elements
        """
        predictions = self.predict(X_test)

        return sum(predictions == y_test) / len(y_test)

iris = load_iris()

X = iris.data
y = iris.target
X, X_test, y, y_test = train_test_split(X, y, test_size= 0.20)

my_tree = MyDecisionTreeClassifier(100)
my_tree.fit(X, y)
print(my_tree.evaluate(X_test, y_test))

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
