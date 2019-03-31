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

# allow command line options
import argparse
parser = argparse.ArgumentParser(description="Perform the simple Plug-in Rule classification on 2-dimensional data provided. (Note: 2D files include 3 columns: x, y, class)")
parser.add_argument("-c", "--classes", type=int, choices=[2], default=2, help="number of class type classifications")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=0, help="increase output verbosity")
args = parser.parse_args()

# compute Euclidean distance between any two given vectors with any length.
# Note: adapted from CSC 587 Adv Data Mining, HW02
# Note: a return value < 0 = Error
def EuclideanDistanceBetweenTwoVectors(vOne, vTwo):
    distance = 0
    v_one_len = len(vOne)
    v_two_len = len(vTwo)
    # vOne & vTwo must be of equal length
    if(v_one_len != v_two_len):
        return -1

    for p in range(0, v_one_len):
        distance += math.pow((abs(vOne[p] - vTwo[p])), 2)
    return math.sqrt(distance)

# variables that are useful to pass around
variables_dict = {
    'training_file' : "./data/homework_classify_train_2D.dat"
    , 'testing_file' : "./data/homework_classify_test_2D.dat"
}

# create a dictionary of list objects equal to the number of class types
class_types_dict = {}
for i in range(1, args.classes + 1):
    class_types_dict[i] = {'count' : 0, 'sum_x' : 0, 'sum_y' : 0, 'mean_x' : -1, 'mean_y' : -1}

if args.verbosity > 0:
    print(f"classes={args.classes} : len(class_types_dict)={len(class_types_dict)}")

# Read the provided data files
# Note: These files are not in a generic (',' or '\t') delimited format -- they require parsing
def ReadFileDataIntoDictOfLists(sFileName, dTraingDict):
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
                    dTraingDict[line_number] = [float(match.group(1)), float(match.group(2)), int(match.group(3))]
                else:
                    print(f"Warning: all three groups were not found on line ({line})")
            else:
                print(f"Warning: no match for line ({line})")
            line_number += 1

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
training_dict = {}
ReadFileDataIntoDictOfLists(variables_dict['training_file'], training_dict)

# Load the testing data
testing_dict = {}
ReadFileDataIntoDictOfLists(variables_dict['testing_file'], testing_dict)

# Print some of the input file data
if args.verbosity > 1:
    print("The first 5 training samples:")
    for i in range(0, len(training_dict)):
        if i > 4:
            break
        else:
            print(f"\t{i} {training_dict[i]}")
    print("The testing samples:")
    for i in range(0, len(testing_dict)):
        print(f"\t{i} {testing_dict[i]}")

GetClassTypeMeans(training_dict, class_types_dict)

print(f"class_types_dict:{class_types_dict}")

prediction_dict = {
    'one_x' : 0
    , 'one_y' : 0
    , 'one_dist' : 0
    , 'two_x' : 0
    , 'two_y' : 0
    , 'two_dist' : 0
}
# loop through all testing data
for i in range(0, len(testing_dict)):
    for j in range(1, len(class_types_dict) + 1):
        if j == 1:
            prediction_dict['one_x'] = (2 * class_types_dict[j]['mean_x'] * testing_dict[i][0]) - (class_types_dict[j]['mean_x'] * class_types_dict[j]['mean_x'])
            prediction_dict['one_y'] = (2 * class_types_dict[j]['mean_y'] * testing_dict[i][1]) - (class_types_dict[j]['mean_y'] * class_types_dict[j]['mean_y'])
            prediction_dict['one_dist'] = EuclideanDistanceBetweenTwoVectors([prediction_dict['one_x'], prediction_dict['one_y']], [class_types_dict[j]['mean_x'], class_types_dict[j]['mean_y']])
        elif j == 2:
            prediction_dict['two_x'] = (2 * class_types_dict[j]['mean_x'] * testing_dict[i][0]) - (class_types_dict[j]['mean_x'] * class_types_dict[j]['mean_x'])
            prediction_dict['two_y'] = (2 * class_types_dict[j]['mean_y'] * testing_dict[i][1]) - (class_types_dict[j]['mean_y'] * class_types_dict[j]['mean_y'])
            prediction_dict['two_dist'] = EuclideanDistanceBetweenTwoVectors([prediction_dict['two_x'], prediction_dict['two_y']], [class_types_dict[j]['mean_x'], class_types_dict[j]['mean_y']])
        else:
            print(f"Warning: class {j} not recognized")

    if prediction_dict['one_dist'] <= prediction_dict['two_dist']:
        if testing_dict[i][2] == 1:
            print(f"Test {i}: predicted class 1 correctly \t| distance from class 1 mean ({round(prediction_dict['one_dist'], 2)}) <= ({round(prediction_dict['two_dist'], 2)}) distance from class 2 mean")
        elif testing_dict[i][2] == 2:
            print(f"Test {i}: predicted class 1 incorrectly \t| distance from class 1 mean ({round(prediction_dict['one_dist'], 2)}) <= ({round(prediction_dict['two_dist'], 2)}) distance from class 2 mean")
        else:
            print(f"Warning 2: class {testing_dict[i][2]} not recognized")
    else:
        if testing_dict[i][2] == 2:
            print(f"Test {i}: predicted class 2 correctly \t| distance from class 2 mean ({round(prediction_dict['two_dist'], 2)}) < ({round(prediction_dict['one_dist'], 2)}) distance from class 1 mean")
        elif testing_dict[i][2] == 1:
            print(f"Test {i}: predicted class 2 incorrectly \t| distance from class 2 mean ({round(prediction_dict['two_dist'], 2)}) < ({round(prediction_dict['one_dist'], 2)}) distance from class 1 mean")
        else:
            print(f"Warning 3: class {testing_dict[i][2]} not recognized")
