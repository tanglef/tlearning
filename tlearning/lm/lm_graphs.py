import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

current_dir = os.path.dirname(os.path.realpath(__file__))
result_dir = os.path.join(current_dir, "static")

##########################
# Simple linear trend
##########################

fig = px.scatter(x=np.linspace(0, 10, num=30),
                 y=np.linspace(0, 10, num=30) + np.random.randn(30),
                 trendline="ols")
fig.write_html(os.path.join(result_dir, "linear_trend.html"), auto_open=False)

######################################
# Non linear trend and transformation
######################################

np.random.seed(11235)
num_noises = 3
noises = [1, 10, 20]
x = np.linspace(0, 2, num=50)
y_noised, y_exp, y_lin = [], [], []
r2_exp, r2_lin = [], []
mse_exp, mse_lin = [], []

for i in range(num_noises):
    y_noised.append(np.exp(np.linspace(0, 2, num=50) + 4) +
                    np.random.normal(size=50, loc=0, scale=noises[i]))
    y_exp.append(np.exp(LinearRegression().fit(x.reshape(-1, 1), np.log(
                 y_noised[i].reshape(-1, 1))).predict(x.reshape(-1, 1))))
    y_lin.append(LinearRegression().fit(x.reshape(-1, 1),
                 y_noised[i].reshape(-1, 1)).predict(x.reshape(-1, 1)))
    r2_exp.append(r2_score(y_exp[i], y_noised[i]))
    r2_lin.append(r2_score(y_lin[i], y_noised[i]))
    mse_exp.append(mean_squared_error(y_exp[i], y_noised[i]))
    mse_lin.append(mean_squared_error(y_lin[i], y_noised[i]))

trace_data = [
    go.Scatter(name="Exponential data: Noise =" + str(noises[i]),
               x=x, y=y_noised[i], mode='markers') for i in range(num_noises)]

trace_exp = [
    go.Scatter(
            x=x,
            y=y_exp[i].reshape(-1),
            mode='lines',
            name='Exponential fit',
            hovertemplate='<b>R2=</b>:' + str(round(r2_exp[i], 2)) + '<br>' +
                          '<b>MSE=</b>:' + str(round(mse_exp[i], 2)) + '<br>'
    ) for i in range(num_noises)]

trace_lin = [
    go.Scatter(
            x=x,
            y=y_lin[i].reshape(-1),
            mode='lines',
            name='Linear fit',
            hovertemplate='<b>R2=</b>:' + str(round(r2_lin[i], 2)) + '<br>' +
                          '<b>MSE=</b>:' + str(round(mse_lin[i], 2)) + '<br>'
    ) for i in range(num_noises)]

fig = go.Figure(data=trace_data + trace_exp + trace_lin)
steps = []
for i in range(num_noises):
    # Hide all traces
    step = dict(method='restyle',
                args=['visible', [False] * len(fig.data)])
    # Enable the two traces we want to see
    step['args'][1][i] = True
    step['args'][1][i+num_noises] = True
    step['args'][1][i+2*num_noises] = True

    # Add step to step list
    steps.append(step)

sliders = [dict(steps=steps,
                active=0,
                currentvalue={"prefix": "Noise: "})]

fig.layout.sliders = sliders
for i, noise in enumerate(noises):
    fig['layout']['sliders'][0]['steps'][i]['label'] = noise
fig.write_html(os.path.join(result_dir, "exp_trend.html"), auto_open=False)
