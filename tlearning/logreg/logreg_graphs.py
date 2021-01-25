import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px


current_dir = os.path.dirname(os.path.realpath(__file__))
result_dir = os.path.join(current_dir, "static")

np.random.seed(11235)

###############################
# Sigmoid
###############################

x = np.array([1, 3.5, 5, 10.75, 15])
total = np.array([50, 60, 50, 30, 40])
deads = np.array([3, 9, 7, 8, 13])


def sigmoid(x):
    return 1/(1+np.exp(-x))


points = np.linspace(-10, 10, num=100)
fig = px.line(x=points,
              y=sigmoid(points),
              title="Sigmoid function",
              labels=dict(x="x", y=r"$\sigma(x)$"))
fig.add_hline(y=.5, line_dash="dot",
              annotation_text="probability=0.5",
              annotation_position="bottom left")
fig.write_html(os.path.join(result_dir, "sigmoid.html"), auto_open=False,
               include_plotlyjs="cdn", include_mathjax='cdn')
