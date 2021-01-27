import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression

current_dir = os.path.dirname(os.path.realpath(__file__))
result_dir = os.path.join(current_dir, "static")

np.random.seed(11235)

###############################
# Sigmoid
###############################

x = np.array([1, 3.5, 5, 10.75, 15])
total = np.array([50, 60, 50, 30, 40])
deads = np.array([3, 10, 15, 20, 32])


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


###########################
# Fitting the model vs LM
###########################

all_y = [[1] * deads[i] + [0] * (total[i] - deads[i]) for i in range(len(x))]
all_y = np.array([y_ for sublist in all_y for y_ in sublist])
all_x = np.repeat(x, total)

points_x = np.linspace(0, 20, num=30)
pchat = deads / total


lr = LinearRegression().fit(all_x.reshape(-1, 1),
                            all_y.reshape(-1, 1))
y_lin = lr.predict(points_x.reshape(-1, 1)).flatten()

logi = LogisticRegression().fit(all_x.reshape(-1, 1),
                                all_y.reshape(-1, 1))
p_logi = sigmoid(points_x * logi.coef_ + logi.intercept_).flatten()

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=pchat, name=r"MLE(H1) p_c",
                         mode="markers"))
fig.add_trace(go.Scatter(x=points_x, y=p_logi, line=dict(color='red'),
                         name=r"logistic model")) # noqa
fig.add_trace(go.Scatter(x=points_x, y=y_lin, name="linear model",
                         line=dict(color="blue")))
fig.update_layout(
    title="Logisitc regression on the smokes",
    xaxis_title="Mean number of smokes per day",
    yaxis_title="probability",
    font=dict(size=13),
    showlegend=True)

fig.write_html(os.path.join(result_dir, "smokers.html"), auto_open=False,
               include_plotlyjs="cdn", include_mathjax='cdn')
