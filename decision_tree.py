import csv
import random
import math
import operator

def read_data(csv_path):
    """Read in the training data from a csv file.
    
    The examples are returned as a list of Python dictionaries, with column names as keys.
    """
    examples = []
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for example in csv_reader:
            for k, v in example.items():
                if v == '':
                    example[k] = None
                else:
                    try:
                        example[k] = float(v)
                    except ValueError:
                         example[k] = v
            examples.append(example)
    return examples


def train_test_split(examples, test_perc):
    """Randomly data set (a list of examples) into a training and test set."""
    test_size = round(test_perc*len(examples))    
    shuffled = random.sample(examples, len(examples))
    return shuffled[test_size:], shuffled[:test_size]


class TreeNodeInterface():
    """Simple "interface" to ensure both types of tree nodes must have a classify() method."""
    def classify(self, example): 
        pass


class DecisionNode(TreeNodeInterface):
    """Class representing an internal node of a decision tree."""

    def __init__(self, test_attr_name, test_attr_threshold, child_lt, child_ge, child_miss):
        """Constructor for the decision node.  Assumes attribute values are continuous.

        Args:
            test_attr_name: column name of the attribute being used to split data
            test_attr_threshold: value used for splitting
            child_lt: DecisionNode or LeafNode representing examples with test_attr_name
                values that are less than test_attr_threshold
            child_ge: DecisionNode or LeafNode representing examples with test_attr_name
                values that are greater than or equal to test_attr_threshold
            child_miss: DecisionNode or LeafNode representing examples that are missing a
                value for test_attr_name                 
        """    
        self.test_attr_name = test_attr_name  
        self.test_attr_threshold = test_attr_threshold 
        self.child_ge = child_ge
        self.child_lt = child_lt
        self.child_miss = child_miss

    def classify(self, example):
        """Classify an example based on its test attribute value.
        
        Args:
            example: a dictionary { attr name -> value } representing a data instance

        Returns: a class label and probability as tuple
        """
        test_val = example[self.test_attr_name]
        if test_val is None:
            return self.child_miss.classify(example)
        elif test_val < self.test_attr_threshold:
            return self.child_lt.classify(example)
        else:
            return self.child_ge.classify(example)

    def __str__(self):
        return "test: {} < {:.4f}".format(self.test_attr_name, self.test_attr_threshold) 


class LeafNode(TreeNodeInterface):
    """Class representing a leaf node of a decision tree.  Holds the predicted class."""

    def __init__(self, pred_class, pred_class_count, total_count):
        """Constructor for the leaf node.

        Args:
            pred_class: class label for the majority class that this leaf represents
            pred_class_count: number of training instances represented by this leaf node
            total_count: the total number of training instances used to build the leaf node
        """    
        self.pred_class = pred_class
        self.pred_class_count = pred_class_count
        self.total_count = total_count
        self.prob = pred_class_count / total_count  # probability of having the class label

    def classify(self, example):
        """Classify an example.
        
        Args:
            example: a dictionary { attr name -> value } representing a data instance

        Returns: a class label and probability as tuple as stored in this leaf node.  This will be
            the same for all examples!
        """
        return self.pred_class, self.prob

    def __str__(self):
        return "leaf {} {}/{}={:.2f}".format(self.pred_class, self.pred_class_count, 
                                             self.total_count, self.prob)


class DecisionTree:
    """Class representing a decision tree model."""

    def __init__(self, examples, id_name, class_name, min_leaf_count=1):
        """Constructor for the decision tree model.  Calls learn_tree().

        Args:
            examples: training data to use for tree learning, as a list of dictionaries
            id_name: the name of an identifier attribute (ignored by learn_tree() function)
            class_name: the name of the class label attribute (assumed categorical)
            min_leaf_count: the minimum number of training examples represented at a leaf node
        """
        self.id_name = id_name
        self.class_name = class_name
        self.min_leaf_count = min_leaf_count

        self.splits = self.make_splits(examples)
        self.set_optimal_splits(examples)
        

        # build the tree!
        self.root = self.learn_tree(examples)  



    def learn_tree(self, examples):
        """Build the decision tree based on entropy and information gain.
        
        Args:
            examples: training data to use for tree learning, as a list of dictionaries.  The
                attribute stored in self.id_name is ignored, and self.class_name is consided
                the class label.
        
        Returns: a DecisionNode or LeafNode representing the tree
        """ 
        # call recursively by simply passing the two partitions into learn_tree as examples
        # splits are based on >=
        
        split = self.calculate_best_key(examples)
        if(len(split[1]) > self.min_leaf_count and len(split[2]) > self.min_leaf_count):
            new_node = DecisionNode(split[0], self.splits[split[0]], self.learn_tree(split[2]), self.learn_tree(split[1]), self.learn_tree(split[3]) if len(split[3]) > 0 else None)
        else:
            pred_class = { 'red': 0, 'light blue': 0, 'medium blue': 0, 'wicked blue': 0 }
            for example in examples:
                pred_class[example['2020_label']] += 1
            key = max(pred_class.items(), key=operator.itemgetter(1))[0]
            new_node = LeafNode(key, pred_class[key], len(examples))
        return new_node

    def calculate_best_key(self, examples):
        parent_entropy = self.calculate_entropy(examples)
        igs = []
        for key in self.splits.keys():
            child_ex1 = []
            child_ex2 = []
            child_ex3 = []
            p1 = 0
            p2 = 0
            p3 = 0
            for example in examples:
                if example[key] == None:
                    child_ex3.append(example)
                    p3 += 1
                elif example[key] >= self.splits[key]:
                    p1 += 1
                    child_ex1.append(example)
                else:
                    child_ex2.append(example)
                    p2 += 1
            p1 /= len(examples)
            p2 /= len(examples)
            p3 /= len(examples)
            e1 = self.calculate_entropy(child_ex1)
            e2 = self.calculate_entropy(child_ex2)
            e3 = self.calculate_entropy(child_ex3)
            ig = parent_entropy - ((p1 * e1) + (p2 * e2) + (p3 * e3))
            igs.append((ig, key, child_ex1, child_ex2, child_ex3))
        max_ig = igs[0]
        for ig in igs:
            if ig[0] > max_ig[0]:
                max_ig = ig
        return (max_ig[1], max_ig[2], max_ig[3], max_ig[4])

    def attribute_entropy(self, examples, attribute, splits):
        max_split = (0, 0)
        for split in splits:
            p = 0
            for example in examples:
                if example[attribute] and example[attribute] >= split:
                    p += 1
            p /= len(examples)
            if p == 0:
                entropy = 0
            else:
                entropy = (p * -1) * math.log2(p)
            if entropy > max_split[1]:
                max_split = (split, entropy)
        return max_split[0]

    def set_optimal_splits(self, examples):
        for key in self.splits.keys():
            val = self.attribute_entropy(examples, key, self.splits[key])
            print(val)
            self.splits[key] = val
    
    def calculate_entropy(self, examples):
        if len(examples) == 0:
            return 0
        entropy = 0
        for key in self.splits.keys():
            p = 0
            for example in examples:
                if example[key] != None and example[key] >= self.splits[key]:
                    p += 1
            p /= len(examples)
            if p != 0:
                entropy += math.log2(p) * (p * -1)    
        return entropy
    
    def make_splits(self, examples):
        #splits = {key1: [ints], key2: [ints], ... }
        splits = {}
        for key in examples[0].keys():
            if key == 'town' or key == '2020_label':
                continue
            values = []
            max_val = None
            min_val = None
            for example in examples:
                if example[key] is None:
                    continue
                if max_val is None:
                    max_val = example[key]
                    min_val = example[key]

                if example[key] < min_val:
                    min_val = example[key]
                elif example[key] > max_val:
                    max_val = example[key]
            values = []
            for i in range(10):
                values.append((i * (max_val - min_val) / 10) + min_val)
            splits[key] = values
        return splits

    def classify(self, example):
        """Perform inference on a single example.

        Args:
            example: the instance being classified

        Returns: a tuple containing a class label and a probability
        """
        return self.root.classify(example)

    def __str__(self):
        """String representation of tree, calls _ascii_tree()."""
        ln_bef, ln, ln_aft = self._ascii_tree(self.root)
        return "\n".join(ln_bef + [ln] + ln_aft)

    def _ascii_tree(self, node):
        """Super high-tech tree-printing ascii-art madness."""
        indent = 7  # adjust this to decrease or increase width of output 
        if type(node) == LeafNode:
            return [""], "leaf {} {}/{}={:.2f}".format(node.pred_class, node.pred_class_count, node.total_count, node.prob), [""]  
        else:
            child_ln_bef, child_ln, child_ln_aft = self._ascii_tree(node.child_ge)
            lines_before = [ " "*indent*2 + " " + " "*indent + line for line in child_ln_bef ]            
            lines_before.append(" "*indent*2 + u'\u250c' + " >={}----".format(node.test_attr_threshold) + child_ln)
            lines_before.extend([ " "*indent*2 + "|" + " "*indent + line for line in child_ln_aft ])

            line_mid = node.test_attr_name
            
            child_ln_bef, child_ln, child_ln_aft = self._ascii_tree(node.child_lt)
            lines_after = [ " "*indent*2 + "|" + " "*indent + line for line in child_ln_bef ]
            lines_after.append(" "*indent*2 + u'\u2514' + "- <{}----".format(node.test_attr_threshold) + child_ln)
            lines_after.extend([ " "*indent*2 + " " + " "*indent + line for line in child_ln_aft ])

            return lines_before, line_mid, lines_after


def confusion4x4(labels, vals):
    """Create an normalized predicted vs. actual confusion matrix for four classes."""
    n = sum([ v for v in vals.values() ])
    abbr = [ "".join(w[0] for w in lab.split()) for lab in labels ]
    s =  ""
    s += " actual ___________________________________  \n"
    for ab, labp in zip(abbr, labels):
        row = [ vals.get((labp, laba), 0)/n for laba in labels ]
        s += "       |        |        |        |        | \n"
        s += "  {:^4s} | {:5.2f}  | {:5.2f}  | {:5.2f}  | {:5.2f}  | \n".format(ab, *row)
        s += "       |________|________|________|________| \n"
    s += "          {:^4s}     {:^4s}     {:^4s}     {:^4s} \n".format(*abbr)
    s += "                     predicted \n"
    return s


#############################################

if __name__ == '__main__':

    path_to_csv = 'town_data.csv'
    class_attr_name = '2020_label'
    id_attr_name = 'town'
    min_examples = 10  # minimum number of examples for a leaf node

    # read in the data
    examples = read_data(path_to_csv)
    train_examples, test_examples = train_test_split(examples, 0.25)

    # learn a tree from the training set
    tree = DecisionTree(train_examples, id_attr_name, class_attr_name, min_examples)

    # test the tree on the test set and see how we did
    correct = 0
    almost = 0  # within one level of correct answer
    ordering = ['red', 'light blue', 'medium blue', 'wicked blue']  # used to count "almost" right
    test_act_pred = {}
    for example in test_examples:
        actual = example[class_attr_name]
        pred, prob = tree.classify(example)
        print("{:30} pred {:15} ({:.2f}), actual {:15} {}".format(example[id_attr_name] + ':', 
                                                            "'" + pred + "'", prob, 
                                                            "'" + actual + "'",
                                                            '*' if pred == actual else ''))
        if pred == actual:
            correct += 1
        if abs(ordering.index(pred) - ordering.index(actual)) < 2:
            almost += 1
        test_act_pred[(actual, pred)] = test_act_pred.get((actual, pred), 0) + 1 

    print("\naccuracy: {:.2f}".format(correct/len(test_examples)))
    print("almost:   {:.2f}\n".format(almost/len(test_examples)))
    print(confusion4x4(['red', 'light blue', 'medium blue', 'wicked blue'], test_act_pred))
    print(tree)  # visualize the tree in sweet, 8-bit text


