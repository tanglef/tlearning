from flask import Blueprint, render_template
from .lm_graphs import *

# Blueprint Configuration
lm_bp = Blueprint(
    'lm_bp', __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/lm/static'
)


@lm_bp.route('/lm/concept_lm', methods=['GET'])
def lm():
    """Linear model page."""

    return render_template(
        'index_lm.html',
        title='LM main page',
        template='lm-template',
    )


@lm_bp.route('/lm/coding_lm', methods=['GET'])
def lm_code():
    """Code linear model page."""

    return render_template(
        'coding_lm.html',
        title='LM code page',
        template='lm-template',
    )
