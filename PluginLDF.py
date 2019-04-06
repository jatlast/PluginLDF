########################################################################
# Jason Baumbach
#   CSC 546 - Homework 8 (due: April 02, 2019 @ 5:59 PM)
#   1.  Plug-in Rule – write the code to perform the simple Plug-in Rule classifier. 
#       Use the same training and test files provided for the k-NN homework.
#
#   - 2-dimensional data: homework_classify_train_2D.dat & homework_classify_test_2D.dat.
#   - (you will need to do vector means and measurements)
#
# Note: this code is available on GitHub 
#   https://github.com/jatlast/PluginLDF.git
#
########################################################################

# required for parsing data files
import re

# allow command line options
import argparse
parser = argparse.ArgumentParser(description="Perform the simple Plug-in Rule classification on 2-dimensional data provided. (Note: 2D files include 3 columns: x, y, class)")
parser.add_argument("-c", "--classes", type=int, choices=[2], default=2, help="number of class type classifications")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=0, help="increase output verbosity")
args = parser.parse_args()

def ReadFileDataIntoMatrix(sFileName, mTrainingData):
    # read in the training data file
    with open(sFileName, mode='r') as data_file:
        line_number = 0
        # parse data file
        for line in data_file:
            # a regular expression to match both the 1-dimensional & 2-dimensional files supplied
            match = re.search(r'(\d+\.\d+)\s+(\d+\.\d+)\s+(\d)', line)
            # get the x (and possibly the y) values
            if match:
                if match.group(1) and match.group(2) and match.group(3):
                    # first, append row
                    mTrainingData.append(line_number)
                    # then, append columns
                    mTrainingData[line_number] = [float(match.group(1)), float(match.group(2)), int(match.group(3))]
                else:
                    print(f"Warning: all three groups were not found on line ({line})")
            else:
                print(f"Warning: no match for line ({line})")
            line_number += 1

# get the inner product (dot product) of two equal length vectors
def GetInnerProductOfTwoVectors(vOne, vTwo):
    product = 0
    v_one_len = len(vOne)
    v_two_len = len(vTwo)
    # vOne & vTwo must be of equal length
    if(v_one_len != v_two_len):
        return -1
    else:
        for i in range(0, v_one_len):
            product += vOne[i] * vTwo[i]
            if args.verbosity > 1:
                print(f"vOne[{i}]:{vOne[i]} * vTwo[{i}]:{vTwo[i]}")
    return product

# calculate the class type means from the training data
def GetClassTypeMeans(dTrainingData, dClassTypes):
    # Loop through the training set to calculate the totals required to calculate the means
    for i in range(0, len(dTrainingData)):
        dClassTypes[dTrainingData[i][2]]['count'] += 1
        dClassTypes[dTrainingData[i][2]]['sum_x'] += dTrainingData[i][0]
        dClassTypes[dTrainingData[i][2]]['sum_y'] += dTrainingData[i][1]

    # calculate the means
    for i in range(1, len(dClassTypes) + 1):
        dClassTypes[i]['mean'] = [dClassTypes[i]['sum_x'] / dClassTypes[i]['count'], dClassTypes[i]['sum_y'] / dClassTypes[i]['count']] 

# variables that are useful to pass around
variables_dict = {
    'training_file' : "./data/homework_classify_train_2D.dat"
    , 'testing_file' : "./data/homework_classify_test_2D.dat"
}

# create a dictionary of list objects equal to the number of class types
class_types_dict = {}
for i in range(1, args.classes + 1):
    class_types_dict[i] = {'count' : 0, 'sum_x' : 0, 'sum_y' : 0, 'mean' : -1}

if args.verbosity > 0:
    print(f"classes={args.classes} : len(class_types_dict)={len(class_types_dict)}")

# Load the training data
training_matrix = []
ReadFileDataIntoMatrix(variables_dict['training_file'], training_matrix)

# Load the testing data
testing_matrix = []
ReadFileDataIntoMatrix(variables_dict['testing_file'], testing_matrix)

# Print some of the input file data
if args.verbosity > 1:
    print("The first 5 training samples:")
    for i in range(0, len(training_matrix)):
        if i > 4:
            break
        else:
            print(f"\t{i} {training_matrix[i]}")
    print("The testing samples:")
    for i in range(0, len(testing_matrix)):
        print(f"\t{i} {testing_matrix[i]}")

# calculate the class means and populate the class_types_dict
GetClassTypeMeans(training_matrix, class_types_dict)

# calculate the inner (dot) products of the two class type means
variables_dict['one_mean_square'] = GetInnerProductOfTwoVectors(class_types_dict[1]['mean'], class_types_dict[1]['mean'])
variables_dict['two_mean_square'] = GetInnerProductOfTwoVectors(class_types_dict[2]['mean'], class_types_dict[2]['mean'])

# debug info
if args.verbosity > 1:
    print(f"one_mean_square:{variables_dict['one_mean_square']} | two_mean_square:{variables_dict['two_mean_square']} | class_types_dict2:{class_types_dict}")

# loop through all testing data and print predictions
for i in range(0, len(testing_matrix)):
    # the test vector
    test_vect = [testing_matrix[i][0], testing_matrix[i][1]]
    # the inner (dot) product of the test vector and class means
    one_dot_mean = GetInnerProductOfTwoVectors(test_vect, class_types_dict[1]['mean'])
    two_dot_mean = GetInnerProductOfTwoVectors(test_vect, class_types_dict[2]['mean'])
    # the final g(x) calculation
    one_g = (2 * one_dot_mean) - variables_dict['one_mean_square']
    two_g = (2 * two_dot_mean) - variables_dict['two_mean_square']

    if args.verbosity > 0:
        print(f"\tone_g(x): {round(one_g, 2)} = (2 * {round(one_dot_mean, 2)}) - {round(variables_dict['one_mean_square'], 2)}")
        print(f"\ttwo_g(x): {round(two_g, 2)} = (2 * {round(two_dot_mean, 2)}) - {round(variables_dict['two_mean_square'], 2)}")

    # print the prediction correctness
    print(f"Test {i}) class {testing_matrix[i][2]}: predicted ", end='')
    if one_g <= two_g:
        print("class 1 ", end='')
        if testing_matrix[i][2] == 1:
            print("correctly \t| ", end='')
        elif testing_matrix[i][2] == 2:
            print("incorrectly\t| ", end='')
        else:
            print(f"Warning 2: class {testing_matrix[i][2]} not recognized")
        print(f"one_g ({round(one_g, 2)}) <= ({round(two_g, 2)}) two_g")
    else:
        print("class 2 ", end='')
        if testing_matrix[i][2] == 2:
            print("correctly \t| ", end='')
        elif testing_matrix[i][2] == 1:
            print("incorrectly\t| ", end='')
        else:
            print(f"Warning 3: class {testing_matrix[i][2]} not recognized")
        print(f"one_g ({round(one_g, 2)}) >= ({round(two_g, 2)}) two_g")
