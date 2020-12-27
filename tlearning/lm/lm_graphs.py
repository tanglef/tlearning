import numpy as np
import os
import plotly.express as px

current_dir = os.path.dirname(os.path.realpath(__file__))
result_dir = os.path.join(current_dir, "static")

fig = px.scatter(x=np.linspace(0, 10, 30),
                 y=np.linspace(0, 10, 30) + np.random.randn(30),
                 trendline="ols")
fig.write_html(os.path.join(result_dir, "linear_trend.html"), auto_open=False)
