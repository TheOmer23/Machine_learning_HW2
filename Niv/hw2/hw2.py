import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = data[:, -1]
    unique_classes, count_classes = np.unique(labels, return_counts=True)
    num_labels = labels.shape[0]
    class_prob = count_classes / num_labels
    gini = 1 - np.sum(class_prob ** 2)   
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = data[:, -1]
    unique_classes, count_classes = np.unique(labels, return_counts=True)
    num_labels = labels.shape[0]
    class_prob = count_classes / num_labels
    entropy = -np.sum(class_prob * np.log2(class_prob))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        labels = self.data[:, -1]
        unique_classes, count_classes = np.unique(labels, return_counts=True)
        
        pred = unique_classes[np.argmax(count_classes)]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.children.append(node)
        self.children_values.append(val)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        node_samples_size = self.data.shape[0]
        proportion_of_node = node_samples_size / n_total_sample
        
        goodness, _ = self.goodness_of_split(self.feature)
        
        self.feature_importance = proportion_of_node * goodness
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        data_size = self.data.shape[0]
        new_impurity = 0
        
        if self.gain_ratio == False:
            origin_impurity = self.impurity_func(self.data)
            feature_values = np.unique(self.data[:,feature])
            
            for value in feature_values:
                groups[value] = self.data[self.data[:,feature] == value]
                group_proportion = groups[value].shape[0] / data_size
                group_impurity = self.impurity_func(groups[value])
                new_impurity += group_proportion * group_impurity
            
            goodness = origin_impurity - new_impurity
            
        else:
            origin_impurity = calc_entropy(self.data)
            split_information = 0
            
            feature_values = np.unique(self.data[:,feature])
            for value in feature_values:
                groups[value] = self.data[self.data[:,feature] == value]
                group_proportion = groups[value].shape[0] / data_size
                group_impurity = calc_entropy(groups[value])
                new_impurity += group_proportion * group_impurity
                
                split_information -= group_proportion * np.log2(group_proportion)
            
            information_gain = origin_impurity - new_impurity
            
            if len(feature_values) == 1:
                goodness = 0
            else:
                goodness = information_gain / split_information  
            
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        labels = self.data[:,-1]
        unique_classes, num_classes = np.unique(labels, return_counts=True)
        if len(unique_classes) == 1:
            self.terminal = True
            return
        
        if self.terminal or self.depth >= self.max_depth:
            self.terminal = True
            return
            
        best_feature = None
        best_feature_goodness = -1
        best_feature_groups = None
        
        for check_feature in range(self.data.shape[1] - 1):
            check_goodness, check_groups = self.goodness_of_split(check_feature)
            if check_goodness > best_feature_goodness:
                best_feature = check_feature
                best_feature_goodness = check_goodness
                best_feature_groups = check_groups
        
        if len(best_feature_groups) == 1:
            self.terminal = True
            return
        
        if self.chi != 1:
            num_of_classes = len(unique_classes)
            num_attributes = len(best_feature_groups)
            degree_of_freedom = (num_attributes - 1) * (num_of_classes - 1) #(num of attrabutes -1) * (num of classes -1)
            chi_square_value = chi_table[degree_of_freedom][self.chi]
            
            p_y_list = []
            for i in range(len(unique_classes)):
                p_y_list.append(num_classes[i] / labels.shape[0])

            x_2 = 0
            
            for feature_value, data_subset in best_feature_groups.items():
                d_f = data_subset.shape[0]
                group_labels = data_subset[:,-1]
                
                p_f_list = []
                e_list = []
                
                for i in range(len(unique_classes)):
                    p_f_list.append(np.count_nonzero(group_labels == unique_classes[i]))
                    e_list.append(d_f * p_y_list[i])                    
                                
                for i in range(len(unique_classes)):
                    x_2 += (p_f_list[i] - e_list[i])**2 / e_list[i]
                                
                
            if x_2 <= chi_square_value:
                self.terminal = True
                return
        
        
        self.feature = best_feature
        for feature_value, data_subset in best_feature_groups.items():
            new_child_node = DecisionNode(data=data_subset, impurity_func=self.impurity_func, depth=self.depth+1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(new_child_node, feature_value)
            
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

                    
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
        ##
        self.depth = 0
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.root = DecisionNode(data=self.data, impurity_func=self.impurity_func ,depth=0, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)        
        nodes_stack = [self.root]
        
        while nodes_stack:
            this_node = nodes_stack.pop(0)
            if this_node.depth > self.depth:
                self.depth = this_node.depth
            this_node.split()
            if this_node.terminal == False:
                this_node.calc_feature_importance(self.data.shape[0])
            nodes_stack.extend(this_node.children)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        this_node = self.root
        while not this_node.terminal:
            this_feature_value = instance[this_node.feature]
            if this_feature_value in this_node.children_values:
                next_child_index = this_node.children_values.index(this_feature_value)
                this_node = this_node.children[next_child_index]
            else:
                break
                
        node = this_node
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        num_instances = dataset.shape[0]
        num_correct = 0
        
        for instance in dataset:
            correct_label = instance[-1]
            tree_pred_label = self.predict(instance)
            
            if correct_label == tree_pred_label:
                num_correct += 1
        
        accuracy = num_correct / num_instances
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
        
    def depth(self):
        return self.root.depth()
        
    def tree_depth(self):
        return self.depth

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        this_tree = DecisionTree(data=X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=True)
        this_tree.build_tree()
        
        this_train_accuracy = this_tree.calc_accuracy(X_train)
        training.append(this_train_accuracy)
        
        this_val_accuracy = this_tree.calc_accuracy(X_validation)
        validation.append(this_val_accuracy)        
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    
    for chi_val in chi_values:
        this_tree = DecisionTree(data=X_train, impurity_func=calc_entropy, chi=chi_val, gain_ratio=True)
        this_tree.build_tree()
        
        this_train_accuracy = this_tree.calc_accuracy(X_train)
        chi_training_acc.append(this_train_accuracy)
        
        this_val_accuracy = this_tree.calc_accuracy(X_test)
        chi_validation_acc.append(this_val_accuracy)
        
        depth.append(this_tree.tree_depth())
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    n_nodes = 1
    
    for child in node.children:
        n_nodes += count_nodes(child)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






