import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

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
                                all_y.ravel())
p_logi = sigmoid(points_x * logi.coef_ + logi.intercept_).flatten()

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=pchat, name=r"MLE(H1) p_c",
                         mode="markers"))
fig.add_trace(go.Scatter(x=points_x, y=p_logi, line=dict(color='red'),
                         name=r"logistic model"))  # noqa
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


###########################
# Logreg with 2 variables
###########################

SEED = 11235
margin, mesh_size = .6, .3
X, y = make_classification(n_samples=100,
                           n_features=2,
                           n_redundant=0,
                           n_repeated=0,
                           n_clusters_per_class=1,
                           class_sep=.8,
                           random_state=SEED)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.3,
                                                    random_state=SEED)


# Create a mesh grid on which we will run our model
x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

# Create classifier, run predictions on grid
clf = LogisticRegression()
clf.fit(X_train, y_train)
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

trace_specs = [
    [X_train, y_train, 0, 'Train', 'square', "#000000", True],
    [X_train, y_train, 1, 'Train', 'circle', "#f75e5e", True],
    [X_test, y_test, 0, 'Test', 'square', "#000000", False],
    [X_test, y_test, 1, 'Test', 'circle', "#f75e5e", False]
]

fig = go.Figure(data=[
    go.Scatter(
        x=X[y == label, 0], y=X[y == label, 1],
        name=f'Label {label}',
        mode='markers', marker_symbol=marker,
        marker_color=col, marker_size=12,
        showlegend=legend
    )
    for X, y, label, _, marker, col, legend in trace_specs
])

fig.add_trace(
    go.Contour(
        x=xrange,
        y=yrange,
        z=Z,
        showscale=True,
        opacity=0.4,
        name='Score',
        hoverinfo='skip',
        colorbar=dict(nticks=10, ticks='outside',
                      ticklen=5, tickwidth=1,
                      showticklabels=True,
                      tickangle=0, tickfont_size=12,
                      title="probability to belong in class 1",
                      titleside='right',
                      titlefont=dict(size=16,
                                     family='Arial, sans-serif')

                      )
    )
)

fig.update_layout(legend_orientation='h',
                  title='Classification with logistic regression using two features.')  # noqa
fig.write_html(os.path.join(result_dir, "binary_classification.html"),
               auto_open=False,
               include_plotlyjs="cdn", include_mathjax='cdn')


###########################
# Binary CE loss
###########################


prob = np.linspace(1e-4, 1-1e-4, num=100, endpoint=True)
loss_y_1 = - np.log(prob)
loss_y_0 = - np.log(1-prob)

fig = go.Figure()

fig.add_trace(go.Scatter(x=prob, y=loss_y_1, name=r"y=1"))
fig.add_trace(go.Scatter(x=prob, y=loss_y_0, name=r"y=0"))

fig.update_layout(title_text="Cross entropy loss",
                  xaxis_title="probability to belong in 1-y_i",
                  yaxis_title=r"$l(y_i, x_i)$")
fig.write_html(os.path.join(result_dir, "CE_loss.html"), auto_open=False,
               include_plotlyjs="cdn", include_mathjax='cdn')
