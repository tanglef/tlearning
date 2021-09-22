import numpy as np
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import torchvision
import dash_table
import sys
import os
import torch

current = os.path.dirname(__file__)
path_utils = os.path.join(current, "..", "utils")
path_data = os.path.join(path_utils, "data")
sys.path.append(os.path.join(path_utils, "character_simulator"))
sys.path.append(os.path.join(path_utils))

import character_simulator as chara  # noqa
from cifar10h import CIFAR10H  # noqa

url_base_pathname = '/dashapp_character/'


def dash_application():
    transform = torchvision.transforms.ToTensor()
    cifar = CIFAR10H(root=path_data, download=True,
                     train=True, transform=transform)

    all_chara = {
        "warrior ⚔️🛡️": {"duel": 0, "brawl": 0},
        "citizen 👨‍👨‍👦👩‍👩‍👧": {"influenced": 0},
        "villain 🦹👹": {"liar": 0},
        "wizard 🧙🦄": {"seer": 0, "pressured": 0},
    }

    dict_of_df = {k: pd.DataFrame(v, index=[0]) for k, v in all_chara.items()}
    df = pd.concat(dict_of_df, axis=1).T
    df = df.reset_index().rename(
        columns={"level_0": "character", "level_1": "strategy", 0: "choice"}
    )
    df.loc[df["character"].duplicated(), "character"] = ""

    FONT_AWESOME = "https://use.fontawesome.com/releases/v5.13.0/css/all.css"
    app = dash.Dash(
        __name__,
        url_base_pathname=url_base_pathname,
        server=False,
        external_stylesheets=[dbc.themes.LUMEN, FONT_AWESOME],
        suppress_callback_exceptions=True,
    )

    home_layout = dbc.Container(
        [
            html.H1("Crowd simulator: different strategies and profiles"),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        children=[
                            dbc.Input(
                                id="input-index",
                                placeholder="Index between 0 and 9999",
                            ),
                            html.Div(id="true-label"),
                            dcc.Graph(id="image-cifar",
                                      style={"margin-bottom": "0px"}),
                        ],
                        width=4,
                    ),
                    html.Div(id="current-number", style={"display": "none"}),
                    dbc.Col(
                        dash_table.DataTable(
                            id="table_chara",
                            data=df.to_dict("records"),
                            columns=[
                                {
                                    "name": i,
                                    "id": i,
                                    "editable": False,
                                    "type": "text",
                                }
                                if i != "choice"
                                else {
                                        "name": i,
                                        "id": i,
                                        "editable": True,
                                        "type": "numeric",
                                }
                                for i in df.columns
                            ],
                            style_cell={"textAlign": "left", 'fontSize': 20},
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "rgb(248, 248, 248)",
                                }
                            ],
                            style_header={
                                "backgroundColor": "rgb(230, 230, 230)",
                                "fontWeight": "bold",
                            },
                            tooltip_conditional=[
                                {
                                    "if": {
                                        "filter_query": '{strategy} contains "duel" or {strategy} contains "brawl"'
                                    },
                                    "type": "markdown",
                                    "value": """A **warrior** is a **spammer**.
                                            Their decision is made with the strategy:\n
    P[i,j] = P(ŷ=i|y=j) = 1(i=c) or 1/K \n\n
    depending if strategy is **duel** or **brawl**.
    """,
                                },
                                {
                                    "if": {
                                        "filter_query": '{strategy} contains "liar"'
                                    },
                                    "type": "markdown",
                                    "value": """A **villain** is an **adversarial attack**.
                                            Their decision is made with the strategy:\n
    P[i,j] = P(ŷ=i|y=j) = 1/(K-1) 1(i != j).
    """,
                                },
                                {
                                    "if": {
                                        "filter_query": '{strategy} contains "seer" or {strategy} contains "pressured"'
                                    },
                                    "type": "markdown",
                                    "value": """A **wizard** is an **expert**.
                                            Their decision is made with the strategy:\n
    P[i,j] = P(ŷ=i|y=j) = 1(i=j).\n\n If an oracle or **seer**, their decision is made to be right.
    But if the council (the others) vote in majority another results and they feel
    **pressured**, they agree with the majority.
    """,
                                },
                                {
                                    "if": {
                                        "filter_query": '{strategy} contains "influenced"'
                                    },
                                    "type": "markdown",
                                    "value": """A **citizen** is any **individual**.
                                            Their decision is made with a strategy
    with hesitations.""",
                                },
                            ],
                            tooltip_delay=0,
                            tooltip_duration=None,
                        ),
                        style={"margin-top": '50px'}),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        children=[
                            html.Label(
                                "Joint probability distribution for citizens confusions",
                                style={"margin-top": '-50px', "z-index": "1000000"}),
                            html.P("Sum of lines must be one and symmetric",
                                   id="alert-matrix", style={"display": "none"}),
                            dash_table.DataTable(
                                id="citizen-matrix",
                                data=pd.DataFrame(
                                    np.eye(10),
                                    index=list(cifar.classes_labels),
                                    columns=list(cifar.classes_labels),
                                ).to_dict("records"),
                                style_cell={"textAlign": "center"},
                                columns=[{
                                    "name": i,
                                    "id": i,
                                    "type": "numeric"
                                } for i in list(cifar.classes_labels)],
                                editable=True,
                                style_data_conditional=[
                                    {
                                        "if": {"row_index": "odd"},
                                        "backgroundColor": "rgb(248, 248, 248)",
                                    },
                                ],
                            ),
                        ],
                        width=3.5,
                    ),
                    dbc.Col(children=[dcc.Graph(id="barplot-chara",
                            style={"margin-top": "-500px", "margin-left": "50px"}),
                    ], width={"size": 9, "offset": 3}),
                ]
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
                "Try and play with the population of the village", className="lead"
            ),
            dbc.Nav(
                [
                    dbc.Button(
                        "Home",
                        id="home",
                        className="ml-auto",
                        href='https://tlearning.herokuapp.com/'
                    )

                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    content = html.Div(id="page-content", style=CONTENT_STYLE)

    app.layout = html.Div(
        [dcc.Location(id="url"), sidebar, content])

    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        if pathname == url_base_pathname:
            return home_layout
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
        [Output("image-cifar", "figure"), Output("true-label", "children")],
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
        fig.update_layout(margin=dict(l=10, r=10, b=10, t=10),
                          height=400, width=400)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        y = torch.Tensor([cifar.true_targets[index]]).type(torch.int)

        return (fig, "The true label is {}".format(cifar.classes_labels[y.item()]))

    @ app.callback(
        Output("alert-matrix", "style"),
        Input("citizen-matrix", "data")
    )
    def check_matrix(data):
        if all([sum(list(data[i].values())) == 1 for i in range(10)]):
            return {"display": "none"}
        else:
            return {"display": "block"}

    @ app.callback(
        Output("barplot-chara", "figure"),
        [Input("table_chara", "data"), Input("current-number", "children"),
         Input("citizen-matrix", "data")],
    )
    def get_votes(df, index, matrix):
        y = torch.Tensor([cifar.true_targets[index]]).type(torch.int)
        ll_chara = pd.DataFrame(
            {
                "character": np.repeat(
                    [
                        "warrior-duel",
                        "warrior-brawl",
                        "citizen-influenced",
                        "villain-liar",
                        "wizard-pressured",
                        "wizard-seer",
                    ],
                    10,
                ),
                "category": cifar.classes_labels * 6,
                "count": [0] * 60,
            }
        )
        for charac in df:
            for n in range(charac["choice"]):
                strat = charac["strategy"]
                if strat == "duel":
                    character = "warrior-duel"
                    people = chara.warrior.Warrior(
                        10, strategy="duel", selection=1)
                    vote = people.answer(y)
                elif strat == "brawl":
                    character = "warrior-brawl"
                    people = chara.warrior.Warrior(
                        10, strategy="brawl", selection=1
                    )
                    vote = people.answer(y)
                elif strat == "influenced":
                    character = "citizen-influenced"
                    people = chara.citizen.Citizen(
                        10, strategy=pd.DataFrame(matrix).to_numpy())  # TODO
                    vote = people.answer(y)
                elif strat == "liar":
                    character = "villain-liar"
                    people = chara.villain.Villain(10)
                    vote = people.answer(y)
                elif strat in ["seer", "pressured"]:
                    if strat == "pressured":
                        character = "wizard-pressured"
                        people = chara.wizard.Wizard(10, strategy="pressured")
                        others = (
                            ll_chara.loc[ll_chara.character !=
                                         "wizard-pressured"]
                            .groupby("category")
                            .sum()["count"]
                        )
                        others = torch.Tensor(
                            others.reindex(cifar.classes_labels).to_numpy()
                        ).reshape(1, 10)
                        vote = people.answer(y, others_result=others)
                    else:
                        character = "wizard-seer"
                        people = chara.wizard.Wizard(10, strategy="seer")
                        vote = people.answer(y)
                ll_chara.loc[
                    (ll_chara.character == character)
                    & (ll_chara.category == cifar.classes_labels[vote]),
                    "count",
                ] += 1
        fig = px.bar(
            ll_chara,
            x="category",
            y="count",
            hover_data=["character"],
            color="character",
            title="Results by character",
        )
        return fig
    return app
