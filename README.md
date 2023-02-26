# Objectives

The learning objectives of this assignment are to:
1. implement feed-forward prediction for a single layer neural network 
2. implement training via back-propagation for a single layer neural network 
 
We will implement a simple single-layer neural network with sigmoid activations
everywhere.
This will include making predictions with a network via forward-propagation, and
training the network via gradient descent, with gradients calculated using
back-propagation.
# Environment Setup

* [Python (version 3.8 or higher)](https://www.python.org/downloads/)
* [numpy](http://www.numpy.org/)
* [pytest](https://docs.pytest.org/)
* [pytest-timeout](https://pypi.org/project/pytest-timeout/)


# Tests

The tests in `test_nn.py` check that each method behaves as expected.
To run all the provided tests, run ``pytest`` from the directory containing
``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
...
collected 5 items

test_nn.py FFFFF                                                         [100%]

=================================== FAILURES ===================================
...
============================== 5 failed in 0.65s ===============================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
...
collected 5 items

test_nn.py .....                                                         [100%]

============================== 5 passed in 0.47s ===============================
```

# Acknowledgments

The author of the test suite (test_nn.py) is Dr. Steven Bethard.
