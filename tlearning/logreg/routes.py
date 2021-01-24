from flask import Blueprint, render_template
from .logreg_graphs import *

# Blueprint Configuration
logreg_bp = Blueprint(
    'logreg_bp', __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/logreg/static'
)


@logreg_bp.route('/logreg/concept_logreg', methods=['GET'])
def logreg():
    """Logistic model page."""

    return render_template(
        'index_logreg.html',
        title='logreg main page',
        template='logreg-template',
    )


# @logreg_bp.route('/logreg/coding_logreg', methods=['GET'])
# def logreg_code():
#     """Code logistic regression page."""

#     return render_template(
#         'coding_logreg.html',
#         title='Logreg code page',
#         template='logreg-template',
#     )


# @logreg_bp.route('/logreg/playing_logreg', methods=['GET'])
# def logreg_play():
#     """Point and click logistic model page."""

#     return render_template(
#         'playing_logereg.html',
#         title='Logreg GUI play page',
#         template='logreg-template',
#     )


# @logreg_bp.route('/logreg/math_logreg', methods=['GET'])
# def logreg_math():
#     """Some maths for the logistic model page."""

#     return render_template(
#         'math_logreg.html',
#         title='logreg math page',
#         template='logreg-template',
#     )
