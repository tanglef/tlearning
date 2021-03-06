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

np.random.seed(11235)

x = np.arange(0, 11, 1)
y = 120 + x * 100
noise = np.random.normal(size=11, loc=0, scale=70)

fig = px.scatter(x=x,
                 y=y + noise,
                 trendline="ols", title="Buying price with noise")
fig.write_html(os.path.join(result_dir, "linear_trend.html"), auto_open=False)


##################################
# Video games L2 minimization
##################################
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y+noise, name="price", mode="markers"))
lr = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
y_lin = lr.predict(x.reshape(-1, 1)).flatten()
fig.add_trace(go.Scatter(x=x, y=y_lin, name="linear model",
                         line=dict(color="blue")))

for i in range(11):
    fig.add_shape(type="line",
                  x0=x[i], y0=y_lin[i], x1=x[i], y1=y[i]+noise[i], xref='x',
                  yref='y', line=dict(color="red", width=3)
                  )

fig.update_shapes(dict(xref='x', yref='y'))
fig.update_layout(
    title="Buying price",
    xaxis_title="Number of rare games amongst the 10",
    yaxis_title="Price you get",
    font=dict(size=13),
    showlegend=False)

fig.write_html(os.path.join(result_dir, "l2_games.html"), auto_open=False)


##########################
# Games sells
##########################

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, name="price"))
fig.add_shape(type="line",
              x0=5, y0=0, x1=5, y1=620, xref='x', yref='y',
              line=dict(color="RoyalBlue", width=3, dash="dashdot")
              )
fig.add_shape(type="line",
              x0=0, y0=620, x1=5, y1=620, xref='x', yref='y',
              line=dict(color="RoyalBlue", width=3, dash="dashdot")
              )
fig.add_trace(go.Scatter(
    x=[5], y=[950],
    text=["<b>5 rares game sell<br> at $620!</b>"],
    mode="text"
))
fig.update_shapes(dict(xref='x', yref='y'))
fig.update_layout(
    title="Buying price",
    xaxis_title="Number of rare games amongst the 10",
    yaxis_title="Price you get",
    font=dict(size=13),
    showlegend=False)

fig.write_html(os.path.join(result_dir, "video_games_1.html"), auto_open=False)


######################################
# Non linear trend and transformation
######################################

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
                                        y_noised[i].reshape(-1, 1)).predict(
                                            x.reshape(-1, 1)))
    r2_exp.append(r2_score(y_exp[i], y_noised[i]))
    r2_lin.append(r2_score(y_lin[i], y_noised[i]))
    mse_exp.append(mean_squared_error(y_exp[i], y_noised[i]))
    mse_lin.append(mean_squared_error(y_lin[i], y_noised[i]))

trace_data = [
    go.Scatter(name="Exponential data: Noise =" + str(noises[i]),
               visible=[True, False, False][i],
               x=x, y=y_noised[i], mode='markers') for i in range(num_noises)]

trace_exp = [
    go.Scatter(
        x=x,
        y=y_exp[i].reshape(-1),
        mode='lines',
        name='Exponential fit',
        visible=[True, False, False][i],
        hovertemplate='<b>R2=</b>:' + str(round(r2_exp[i], 2)) + '<br>' +
        '<b>MSE=</b>:' + str(round(mse_exp[i], 2)) + '<br>'
    ) for i in range(num_noises)]

trace_lin = [
    go.Scatter(
        x=x,
        y=y_lin[i].reshape(-1),
        mode='lines',
        name='Linear fit',
        visible=[True, False, False][i],
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
