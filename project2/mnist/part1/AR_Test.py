import sys
import numpy as np
import matplotlib.pyplot as plt

from mnist.part1.linear_regression import compute_test_error_linear
from mnist.part1.softmax import compute_probabilities
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *

x = np.array([[1,1,1],[1,0,1]])
large_theta = np.array([[9999, 9999, 9999], [5555, 5555, 5555], [0, 0, 0]])
temp = [0.5,1,999999999999]

for temp_parameter in temp:
    H = compute_probabilities(x, large_theta, temp_parameter)
    print(f'for temp = {temp_parameter}')
    print(H)