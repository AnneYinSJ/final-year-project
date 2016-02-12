import sys
import pandas as pd
import math
"""
Suit1 is first best splitting attribute.
Suit3 is second best.
"""
def get_file():
    """
    Tries to extract a filename from the command line.  If none is present, it
    prompts the user for a filename and tries to open the file.  If the file
    exists, it returns it, otherwise it prints an error message and ends
    execution.
    """
    # Get the name of the data file and load it into
    if len(sys.argv) < 2:
        # Ask the user for the name of the file
        print "Filename: ",
        filename = sys.stdin.readline().strip()
    else:
        filename = sys.argv[1]

    try:
        fin = open(filename, "r")
    except IOError:
        print "Error: The file '%s' was not found on this system." % filename
        sys.exit(0)

    return fin

def entropy(data, target_attr):
    """
    Calculates the entropy of the given data set for the target attribute.
    """
    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for i in data.index:
        if (val_freq.has_key(data.loc[i][target_attr])):
            val_freq[data.loc[i][target_attr]] += 1.0
        else:
            val_freq[data.loc[i][target_attr]] = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2)
    print "Entropy computed..."
    return data_entropy

def gain(data, attr, target_attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    print "Computing Information Gain..."
    val_freq = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for i in data.index:
        if (val_freq.has_key(data.loc[i][target_attr])):
            val_freq[data.loc[i][target_attr]] += 1.0
        else:
            val_freq[data.loc[i][target_attr]] = 1.0
    print "For loop ran successfully"
    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        print val
        val_prob = val_freq[val] / sum(val_freq.values())
        print val_prob
        #data_subset = [record for record in data if record[attr] == val]
        '''
        for i in data.index:
            print i
            if data.loc[i][target_attr] == val:
                data_subset = data_subset.append(pd.DataFrame(data.loc[i]))
        '''
        data_subset = data.loc[data[attr] == val]
        print len(data_subset)
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    print "Information gain computed..."
    return (entropy(data, target_attr) - subset_entropy)

def choose_attribute(data, attributes, target_attr):
    """
    Cycles through all the attributes and returns the attribute with the
    highest information gain (or lowest entropy).
    """
    data = data
    best_gain = 0.0
    best_attr = None
    print "Choosing best attribute..."
    for attr in attributes:
        print "In here..."
        info_gain = gain(data, attr, target_attr)
        if (info_gain >= best_gain and attr != target_attr):
            best_gain = info_gain
            best_attr = attr
    print best_attr+" chosen as best attribute..."
    return best_attr

def majority_value(data, target_attr):
    """
    Creates a list of all values in the target attribute for each record
    in the data list object, and returns the value that appears in this list
    most frequently.
    """
    list = data.loc[:,target_attr]
    return list.mode()[0]

def get_values(data, attr):
    """
    Creates a list of unique values in the chosen attribute for the data and returns the list.
    """
    data = data
    return pd.unique(data.loc[:,attr]).tolist()

def get_examples(data, attr, value):
    """
    Returns a list of all the records in data with the value of attribute attr
    matching the given value.
    """
    print "Getting examples..."
    data = data
    rtn_list = pd.DataFrame()
    if data.empty:
        return rtn_list
    else:
        rtn_list = data.loc[data[attr] == value]
        print "Obtained examples..."
        return rtn_list

def create_decision_tree(data,attributes,target_attr):
    print "Inside create method..."
    data = data
    vals = [data.loc[i][target_attr] for i in data.index]
    default = majority_value(data, target_attr)

    if data.empty or (len(attributes) - 1) <= 0:
        return default

    elif vals.count(vals[0]) == len(vals):
        return vals[0]

    else:
        best = choose_attribute(data, attributes, target_attr)
        tree = {best:{}}
        for val in get_values(data, best):
            subtree = create_decision_tree(
                get_examples(data, best, val), [attr for attr in attributes if attr != best], target_attr)
            tree[best][val] = subtree

    return tree

def construct_tree(fin):
    print "Tree being constructed..."
    data = pd.read_csv(fin)
    #Giving the columns meaningful names
    data.columns = ["Suit1","Card1","Suit2","Card2","Suit3","Card3","Suit4","Card4","Suit5","Card5","Poker Hand"]
    attributes = [attr for attr in data.columns]
    target_attr = attributes[-1]
    tree = create_decision_tree(data, attributes, target_attr)
    return tree
    '''
    classification = classify(tree,examples)
    for item in classification:
        print item
    return tree
    '''

if __name__ == "__main__":
    #fin = get_file()
    tree = construct_tree(fin = "C:/Users/HP-PC/Desktop/final-year-project/data/poker-hand-training-true.data")

