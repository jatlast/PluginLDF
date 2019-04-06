# PluginLDF
Plugin Linear Discriminant Function

```
Master's Degree: University of Michigan - Computer Science & Information Systems
Course: CSC 546 - Advanced Artificial Intelligence

Assignment Homework 8: Due: 04/09/2019 5:59 PM
    1.  Plug-in Rule – write the code to perform the simple Plug-in Rule classifier. 
       Use the same training and test files provided for the k-NN homework.

    - 2-dimensional data: homework_classify_train_2D.dat & homework_classify_test_2D.dat.
    - (you will need to do vector means and measurements)
```
## False Starts

1) Based on the formula for g(x), I incorrectly assumed 1-dimensional vectors would require transposition.

## Chosen Technologies

Motivation: Become more familiar with the following.
1) Artificial Intelligence supervised training algorithms for classification and prediction
2) IDE - Visual Studio Code

## Caveates

1) Because the goal of homework is to become familiar with the underlying formulas, the resulting code is very non-pythonic.
2) Python is used here more as pseudo-C/C++ code with total disregard for OO-programming.
3) This code is often expanded to more clearly demonstrate understanding of the calculations.

### Prerequisites

- Python 3.6+

### Installing
```
Clone the "PluginLDF" project into the desired directory.
PluginLDF should run in any Python 3.6+ environment
```

### Command Line Specifications
```
> python PluginLDF.py -h
--------------------------------------------------------------------
usage: PluginLDF.py [-h] [-c {2}] [-v {0,1,2}]

Perform the simple Plug-in Rule classification on 2-dimensional data provided.
(Note: 2D files include 3 columns: x, y, class)

optional arguments:
  -h, --help            show this help message and exit
  -c {2}, --classes {2}
                        number of class type classifications
  -v {0,1,2}, --verbosity {0,1,2}
                        increase output verbosity
--------------------------------------------------------------------

Example> python PluginLDF.py -v 2
Results:    -v) verbosity is set to it most verbose setting of 2
            -c) classes is set to 2 because 2 is the default number of classes
```

### Output Examples
```
> python PluginLDF.py
Test 0) class 2: predicted class 1 incorrectly  | one_g (320.43) <= (348.98) two_g
Test 1) class 1: predicted class 1 correctly    | one_g (283.44) <= (310.25) two_g
Test 2) class 1: predicted class 2 incorrectly  | one_g (473.75) >= (367.35) two_g
Test 3) class 1: predicted class 2 incorrectly  | one_g (277.44) >= (251.39) two_g
Test 4) class 2: predicted class 2 correctly    | one_g (205.64) >= (201.16) two_g

> python PluginLDF.py -v 1
        one_g(x): 320.43 = (2 * 276.0) - 231.56
        two_g(x): 348.98 = (2 * 318.92) - 288.87
Test 0) class 2: predicted class 1 incorrectly  | one_g (320.43) <= (348.98) two_g
        one_g(x): 283.44 = (2 * 257.5) - 231.56
        two_g(x): 310.25 = (2 * 299.56) - 288.87
Test 1) class 1: predicted class 1 correctly    | one_g (283.44) <= (310.25) two_g
        one_g(x): 473.75 = (2 * 352.66) - 231.56
        two_g(x): 367.35 = (2 * 328.11) - 288.87
Test 2) class 1: predicted class 2 incorrectly  | one_g (473.75) >= (367.35) two_g
        one_g(x): 277.44 = (2 * 254.5) - 231.56
        two_g(x): 251.39 = (2 * 270.13) - 288.87
Test 3) class 1: predicted class 2 incorrectly  | one_g (277.44) >= (251.39) two_g
        one_g(x): 205.64 = (2 * 218.6) - 231.56
        two_g(x): 201.16 = (2 * 245.01) - 288.87
Test 4) class 2: predicted class 2 correctly    | one_g (205.64) >= (201.16) two_g
```

## License

This project is not licensed but feel free to play with any part you so desire.

## Acknowledgments

* Google's vast doorway to every tid-bit of documentation on the internet
* All those wonderfully generous documentation writers and question answerers
