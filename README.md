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
2) Python is used here more as pseudo-C/C++ code with total disregard for OO-programming

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

## License

This project is not licensed but feel free to play with any part you so desire.

## Acknowledgments

* Google's vast doorway to every tid-bit of documentation on the internet
* All those wonderfully generous documentation writers and question answerers
