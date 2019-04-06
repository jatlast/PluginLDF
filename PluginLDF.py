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

# required for sqrt function in Euclidean Distance calculation
import math
# required for parsing data files
import re
# required for inherent matrix manipulation
#import numpy as np

# allow command line options
import argparse
parser = argparse.ArgumentParser(description="Perform the simple Plug-in Rule classification on 2-dimensional data provided. (Note: 2D files include 3 columns: x, y, class)")
parser.add_argument("-c", "--classes", type=int, choices=[2], default=2, help="number of class type classifications")
#parser.add_argument("-ctc", "--classtypecolumn", type=int, default=3, help="the column number containing the class type classifications")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=0, help="increase output verbosity")
args = parser.parse_args()

# variables that are useful to pass around
variables_dict = {
    'training_file' : "./data/homework_classify_train_2D.dat"
    , 'testing_file' : "./data/homework_classify_test_2D.dat"
#    , 'class_type_column' : args.classtypecolumn - 1
}

# create a dictionary of list objects equal to the number of class types
class_types_dict = {}
for i in range(1, args.classes + 1):
    class_types_dict[i] = {'count' : 0, 'sum_x' : 0, 'sum_y' : 0, 'mean_x' : -1, 'mean_y' : -1}

if args.verbosity > 0:
    print(f"classes={args.classes} : len(class_types_dict)={len(class_types_dict)}")
#    print(f"class_type_column={variables_dict[class_type_column]} : classes={args.classes} : len(class_types_dict)={len(class_types_dict)}")

# Read the provided data files
# Note: These files are not in a generic (',' or '\t') delimited format -- they require parsing
# def ReadFileDataIntoDictOfLists(sFileName, dTraingDict):
#     # read in the training data file
#     with open(sFileName, mode='r') as data_file:
#         line_number = 0
#         # parse data file
#         for line in data_file:
#             # a regular expression to match both the 1-dimensional & 2-dimensional files supplied
#             match = re.search(r'(\d+\.\d+)\s+(\d+\.\d+)\s+(\d)', line)
#             # get the x (and possibly the y) values
#             if match:
#                 if match.group(1) and match.group(2) and match.group(3):
#                     dTraingDict[line_number] = [float(match.group(1)), float(match.group(2)), int(match.group(3))]
#                 else:
#                     print(f"Warning: all three groups were not found on line ({line})")
#             else:
#                 print(f"Warning: no match for line ({line})")
#             line_number += 1

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

# transpose code adapted from:
#   https://www.geeksforgeeks.org/transpose-matrix-single-line-python/
def TransposeMatrix(mOriginal):
    t_matrix = [[mOriginal[j][i] for j in range(len(mOriginal))] for i in range(len(mOriginal[0]))]
#    t_matrix = zip(*mOriginal)
#    t_matrix = map(list, zip(*mOriginal))
    return t_matrix

def GetMatrixShape(vMatrix):
    shape = [0, 0]
    shape[1] = len(vMatrix[0])
    for i in range(0, len(vMatrix)):
        shape[0] += 1
    return shape

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
            if args.verbosity > 0:
                print(f"vOne[{i}]:{vOne[i]} * vTwo[{i}]:{vTwo[i]}")
    return product

# def GetScalarInnerProduct(mLeft, mRight):
#     shape_left = GetMatrixShape(mLeft)
#     shape_right = GetMatrixShape(mRight)
#     scalar = 0
#     # the inner shapes of the two matricies must be equal
#     if shape_left[1] != shape_right[0]:
#         return -1
#     else:
#         for i in range(0, shape_left[0]):
#             for j in range(0, shape_left[0]):
#                 scalar += mLeft[i][j] + mRight[i][j]
#     return scalar

# def GetDictMatrixShape(dMatrix):
#     shape = [0, 0]
#     shape[1] = len(dMatrix[0])
#     for i in range(0, len(dMatrix)):
#         shape[0] += 1
#     return shape

# Get all the class type means
def GetClassTypeMeans(dTrainingData, dClassTypes):
    # Loop through the training set to calculate the totals required to calculate the means
    for i in range(0, len(dTrainingData)):
        dClassTypes[dTrainingData[i][2]]['count'] += 1
        dClassTypes[dTrainingData[i][2]]['sum_x'] += dTrainingData[i][0]
        dClassTypes[dTrainingData[i][2]]['sum_y'] += dTrainingData[i][1]

    # calculate the means
    for i in range(1, len(dClassTypes) + 1):
        dClassTypes[i]['mean_x'] = dClassTypes[i]['sum_x'] / dClassTypes[i]['count']
        dClassTypes[i]['mean_y'] = dClassTypes[i]['sum_y'] / dClassTypes[i]['count']

# Load the training data
training_matrix = []
ReadFileDataIntoMatrix(variables_dict['training_file'], training_matrix)

# Load the testing data
testing_matrix = []
ReadFileDataIntoMatrix(variables_dict['testing_file'], testing_matrix)

# t_testing_matrix = TransposeMatrix(training_matrix)

# if args.verbosity > 1:
#     shape = GetMatrixShape(training_matrix)
#     print(f"training shape:({shape})")
#     shape = GetMatrixShape(t_testing_matrix)
#     print(f"training transposed shape:({shape})")
#     shape = GetMatrixShape(testing_matrix)
#     print(f"testing shape:({shape})")

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

GetClassTypeMeans(training_matrix, class_types_dict)

variables_dict['one_mean_product'] = GetInnerProductOfTwoVectors([class_types_dict[1]['mean_x'], class_types_dict[1]['mean_y']], [class_types_dict[1]['mean_x'], class_types_dict[1]['mean_y']])
variables_dict['two_mean_product'] = GetInnerProductOfTwoVectors([class_types_dict[2]['mean_x'], class_types_dict[2]['mean_y']], [class_types_dict[2]['mean_x'], class_types_dict[2]['mean_y']])

print(f"one_mean_product:{variables_dict['one_mean_product']} | two_mean_product:{variables_dict['two_mean_product']} | class_types_dict2:{class_types_dict}")

prediction_dict = {
    'one_g' : 0
    , 'two_g' : 0
}
# loop through all testing data
for i in range(0, len(testing_matrix)):
    prediction_dict['one_g'] = (2 * GetInnerProductOfTwoVectors([testing_matrix[i][0], testing_matrix[i][1]], [class_types_dict[1]['mean_x'], class_types_dict[1]['mean_y']])) - variables_dict['one_mean_product']
    prediction_dict['two_g'] = (2 * GetInnerProductOfTwoVectors([testing_matrix[i][0], testing_matrix[i][1]], [class_types_dict[2]['mean_x'], class_types_dict[2]['mean_y']])) - variables_dict['two_mean_product']

    # print the prediction correctness
    print(f"Test {i}: predicted ", end='')
    if prediction_dict['one_g'] <= prediction_dict['two_g']:
        print("class 1 ", end='')
        if testing_matrix[i][2] == 1:
            print("correctly \t| ", end='')
        elif testing_matrix[i][2] == 2:
            print("incorrectly \t| ", end='')
        else:
            print(f"Warning 2: class {testing_matrix[i][2]} not recognized")
        print(f"one_g ({round(prediction_dict['one_g'], 2)}) <= ({round(prediction_dict['two_g'], 2)}) two_g")
    else:
        print("class 2 ", end='')
        if testing_matrix[i][2] == 2:
            print("correctly \t| ", end='')
        elif testing_matrix[i][2] == 1:
            print("incorrectly \t| ", end='')
        else:
            print(f"Warning 3: class {testing_matrix[i][2]} not recognized")
        print(f"one_g ({round(prediction_dict['one_g'], 2)}) >= ({round(prediction_dict['two_g'], 2)}) two_g")


    # for j in range(1, len(class_types_dict) + 1):
    #     if j == 1:
    #         prediction_dict['one_x'] = (2 * class_types_dict[j]['mean_x'] * testing_matrix[i][0]) - (class_types_dict[j]['mean_x'] * class_types_dict[j]['mean_x'])
    #         prediction_dict['one_y'] = (2 * class_types_dict[j]['mean_y'] * testing_matrix[i][1]) - (class_types_dict[j]['mean_y'] * class_types_dict[j]['mean_y'])
    #         prediction_dict['one_dist'] = EuclideanDistanceBetweenTwoVectors([prediction_dict['one_x'], prediction_dict['one_y']], [class_types_dict[j]['mean_x'], class_types_dict[j]['mean_y']])
    #     elif j == 2:
    #         prediction_dict['two_x'] = (2 * class_types_dict[j]['mean_x'] * testing_matrix[i][0]) - (class_types_dict[j]['mean_x'] * class_types_dict[j]['mean_x'])
    #         prediction_dict['two_y'] = (2 * class_types_dict[j]['mean_y'] * testing_matrix[i][1]) - (class_types_dict[j]['mean_y'] * class_types_dict[j]['mean_y'])
    #         prediction_dict['two_dist'] = EuclideanDistanceBetweenTwoVectors([prediction_dict['two_x'], prediction_dict['two_y']], [class_types_dict[j]['mean_x'], class_types_dict[j]['mean_y']])
    #     else:
    #         print(f"Warning: class {j} not recognized")
