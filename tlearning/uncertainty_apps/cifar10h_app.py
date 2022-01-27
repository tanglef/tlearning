import sys
import numpy as np
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from dash.dependencies import Input, Output
import torchvision
import os
import json

transform = torchvision.transforms.ToTensor()

current = os.path.dirname(__file__)
path_utils = os.path.join(current, "..", "utils")
path_data = os.path.join(path_utils, "data")
sys.path.append(os.path.join(path_utils, "character_simulator"))
sys.path.append(os.path.join(path_utils))

from cifar10h import CIFAR10H  # noqa

url_base_pathname = "/dashapp_cifar/"


def dash_application():
    cifar = CIFAR10H(
        root=path_data, download=True, train=True, transform=transform
    )

    def spe_index(df, idx):
        raw = df[df.annotator_id == idx]
        raw = raw[(raw.is_attn_check != 1) & (raw.reaction_time > 0)]

        sensi, speci = [], []
        for cat in range(10):
            tp = raw[raw.chosen_label == cat].correct_guess.sum()
            tn = len(raw[(raw.chosen_label != cat) & (raw.true_label != cat)])
            fp = len(raw[(raw.chosen_label == cat) & (raw.true_label != cat)])
            fn = len(raw[(raw.true_label == cat) & (raw.chosen_label != cat)])

            sensi.append(tp / (tp + fn))
            speci.append(tn / (tn + fp))
        return sensi, speci

    def get_spammer_scores(n):
        raw = cifar.get_raw()
        raw = raw[(raw.is_attn_check != 1) & (raw.reaction_time > 0)]

        spam = []
        if n == -1:
            n = len(raw.annotator_id.unique())
        for idx in tqdm(range(n)):
            df = raw[raw.annotator_id == idx]
            A = np.zeros((10, 10))
            for c in range(10):
                raw_c = df[df.true_label == c]
                denom = len(raw_c)
                for k in range(10):
                    num_kc = len(raw_c[raw_c.chosen_label == k])
                    A[c, k] = num_kc / denom
            spam.append(
                1
                / 90
                * np.sum(((A[np.newaxis, :, :] - A[:, np.newaxis, :]) ** 2))
                / 2
            )
        return spam

    def fig_spam(n, ds=False):
        # spam = get_spammer_scores(n)
        # ids = np.arange(len(spam))
        # with open('./data.json', 'w') as fp:
        #     json.dump(dict(x=list(ids.astype("float64")), y=spam), fp)
        file = "data.json" if ds is False else "data_ds.json"
        with open(os.path.join(current, file)) as json_file:
            data = json.load(json_file)
        ids, spam = data["x"], data["y"]
        fig = go.Figure(
            [
                go.Bar(
                    x=ids,
                    y=spam,
                    name="spam score",
                )
            ]
        ).update_layout(
            barmode="stack",
            xaxis={
                "range": [0, 200],
                "rangeslider": {"visible": True},
            },
            margin={"l": 0, "r": 0, "t": 0, "r": 0},
        )
        for wh in np.where(np.array(spam) < 0.5)[0]:
            fig.add_annotation(
                x=wh,
                y=spam[wh],
                xref="x",
                yref="y",
                text="",
                showarrow=True,
                align="center",
                arrowhead=3,
                arrowsize=5,
                arrowwidth=2,
                arrowcolor="red",
                ax=20,
                ay=-30,
            )
        return fig

    app = dash.Dash(
        __name__,
        server=False,
        url_base_pathname=url_base_pathname,
        external_stylesheets=[dbc.themes.LUMEN],
        suppress_callback_exceptions=True,
    )

    home_layout = dbc.Container(
        [
            html.H1("CIFAR10H: uncertainty visualization"),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        children=[
                            dbc.Input(
                                id="input-index",
                                placeholder="Index between 0 and 9999",
                            ),
                        ],
                        width=3,
                    ),
                    html.Div(id="current-number", style={"display": "none"}),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        children=[
                            dcc.Graph(id="image-cifar"),
                            html.Div(id="true-label"),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        children=[
                            html.Div(
                                [
                                    dcc.Tabs(
                                        [
                                            dcc.Tab(
                                                label="Votes",
                                                children=[
                                                    dcc.Graph(id="barplots")
                                                ],
                                            ),
                                            dcc.Tab(
                                                label="Times",
                                                children=[
                                                    dcc.Graph(id="boxplots")
                                                ],
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                    ),
                ],
                justify="center",
            ),
        ],
        fluid=True,
    )

    voters_layout = dbc.Container(
        [
            html.H1("CIFAR10H: uncertainty visualization"),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        children=[
                            dbc.Input(
                                id="idx-voter",
                                placeholder="Index between 0 and 2570",
                            ),
                            html.Button(
                                "Reset barplot",
                                id="submit-spam-fig",
                                n_clicks=0,
                            ),
                        ],
                        width=3,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        children=[
                            html.Div(
                                [dcc.Graph(id="barplots-voter")],
                            ),
                        ],
                        width=5,
                    ),
                    dbc.Col(
                        children=[
                            html.Div(id="index-voter"),
                            html.Div([dcc.Graph(id="cobweb-voter")]),
                        ],
                        width=5,
                    ),
                ],
                justify="center",
            ),
            html.H2("Spam scores using ground truths"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            children=[
                                # html.Iframe(src="./barplot_spam.html",
                                #             style={"max-width:100%; max-height:100%;"})
                                dcc.Graph(id="spam-fig"),
                            ]
                        ),
                    ),
                ],
            ),
            html.H2("Spam scores using DS model"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            children=[
                                # html.Iframe(src="./barplot_spam.html",
                                #             style={"max-width:100%; max-height:100%;"})
                                dcc.Graph(id="spam-fig-ds"),
                            ]
                        ),
                    ),
                ],
            ),
        ],
        fluid=True,
    )

    # from https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/page-2
    # the style arguments for the sidebar. We use position:fixed and a fixed width
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }

    # the styles for the main content position it to the right of the sidebar and
    # add some padding.
    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }

    sidebar = html.Div(
        [
            html.H2("Viewpoint", className="display-4"),
            html.Hr(),
            html.P(
                "Navigate between voters and images perspectives",
                className="lead",
            ),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Images", href=url_base_pathname)),
                    dbc.NavItem(
                        dbc.NavLink(
                            "Voters", href=url_base_pathname + "voters"
                        )
                    ),
                    dbc.Button(
                        "Home",
                        id="home",
                        className="ml-auto",
                        href="https://tlearning.herokuapp.com/",
                    ),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    content = html.Div(id="page-content", style=CONTENT_STYLE)

    app.layout = html.Div(
        [dcc.Location(id="url", refresh=False), sidebar, content]
    )

    @app.callback(
        Output("page-content", "children"), [Input("url", "pathname")]
    )
    def render_page_content(pathname):
        if pathname == url_base_pathname:
            return home_layout
        elif pathname == url_base_pathname + "voters":
            return voters_layout
        elif pathname != "/":
            # If the user tries to reach a different page, return a 404 message
            return dbc.Jumbotron(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognised..."),
                ]
            )

    @app.callback(
        Output("current-number", "children"), Input("input-index", "value")
    )
    def randomize(index):
        if index is None or index == "":
            return np.random.randint(low=0, high=9999, size=1)[0]
        index = int(index)
        if index > 9999:
            return np.random.randint(low=0, high=9999, size=1)[0]
        else:
            return index

    @app.callback(
        [
            Output("barplots", "figure"),
            Output("image-cifar", "figure"),
            Output("boxplots", "figure"),
            Output("true-label", "children"),
        ],
        [Input("current-number", "children")],
    )
    def get_uncertainty(index):
        try:
            im = cifar[index][0]
        except Exception as e:
            print(e)
            return px.scatter(title="Error: " + e)
        im = np.transpose(im.numpy(), (1, 2, 0))
        fig = px.imshow(im)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        distrib_fig = go.Figure(
            [go.Bar(x=cifar.classes_labels, y=cifar[index][1][0].numpy())]
        )

        raw = cifar.get_raw()
        raw = raw[raw.cifar10_test_test_idx == index]
        raw = raw[(raw.is_attn_check != 1) & (raw.reaction_time > 0)]
        boxplots = px.box(raw, x="correct_guess", y="reaction_time")
        return (
            distrib_fig,
            fig,
            boxplots,
            "The true label is {}".format(
                cifar.classes_labels[cifar.true_targets[index]]
            ),
        )

    @app.callback(
        [
            Output("barplots-voter", "figure"),
            Output("cobweb-voter", "figure"),
            Output("index-voter", "children"),
        ],
        [Input("idx-voter", "value")],
    )
    def get_sens_spe(index):
        if index is None or index == "":
            index = np.random.randint(low=0, high=2570, size=1)[0]
        index = int(index)
        if index > 2570 or index < 0:
            index = np.random.randint(low=0, high=2570, size=1)[0]

        raw = cifar.get_raw()
        sensi, speci = spe_index(raw, index)

        bar = go.Figure(
            data=[
                go.Bar(name="sensibility", x=cifar.classes_labels, y=sensi),
                go.Bar(name="specificity", x=cifar.classes_labels, y=speci),
            ]
        )
        # Change the bar mode
        bar.update_layout(barmode="group")

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=sensi,
                theta=cifar.classes_labels,
                fill="toself",
                name="sensibility",
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=speci,
                theta=cifar.classes_labels,
                fill="toself",
                name="specificity",
            )
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
        )
        return (bar, fig, "Looking at voter number {}".format(index))

    @app.callback(
        Output("spam-fig", "figure"), Input("submit-spam-fig", "n_clicks")
    )
    def spam_figure(n_clicks):
        if n_clicks < 1:
            n = 200
        else:
            n = -1
        spam_fig = fig_spam(n)
        return spam_fig

    @app.callback(
        Output("spam-fig-ds", "figure"), Input("submit-spam-fig", "n_clicks")
    )
    def spam_figure_ds(n_clicks):
        if n_clicks < 1:
            n = 200
        else:
            n = -1
        spam_fig = fig_spam(n, ds=True)
        return spam_fig

    return app
