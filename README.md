This is an implementation of the [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)
algorithm as a python extension and simple CLI tool.

In progress but currently works okay on OS X and Python 2.7.

## Dependencies 

### For the CLI tool

* A working C++14 compiler and libc++; Xcode 7+'s tools should be fine on OS X
* Boost's numeric bindings to UBLAS (specifically, `#include
  <boost/numeric/ublas/vector_sparse.hpp>` needs to work on your system)

### For the Python extension

Setuptools is supposed to manage this when you run install, but if it doesnt,
it's the above dependencies plus

* Python 2.7.x (haven't tested Python 3 yet, shouldn't be hard though)
* numpy

For tests

* scikit-learn
* nose

## Build

### For the CLI tool

```make```

### Python

If you like to test things are working, `python setup.py test` then

```python setup.py install```

## Run

### The CLI tool

```
usage: dbscan eps min_pts array_type distance_metric precision input_path
eps:        parameter to dbscan algorithm, e.g. 0.3
min_pts:    parameter to dbscan algorithm, e.g. 10
array_type: can be sparse or nonsparse
distance_metric: can be euclidean or cosine
precision:  can be double or single
input_path: is the path of a CSV containing vectors
```

### The Python Extension

There's a great demo in [plot.py](plot.py) which I've adapted from [one of the
scikit-learn
ones](http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html).
