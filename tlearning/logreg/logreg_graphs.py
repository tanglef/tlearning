import numpy as np
import os
import plotly.graph_objects as go

current_dir = os.path.dirname(os.path.realpath(__file__))
result_dir = os.path.join(current_dir, "static")

np.random.seed(11235)

###############################
# Sigmoid
###############################

x = np.array([1, 3.5, 5, 10.75, 15])
total = np.array([50, 60, 50, 30, 40])
deads = np.array([3, 9, 7, 8, 13])
