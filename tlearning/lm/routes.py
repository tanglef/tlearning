from flask import Blueprint, render_template
from .lm_graphs import *

# Blueprint Configuration
lm_bp = Blueprint(
    'lm_bp', __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/lm/static'
)


@lm_bp.route('/lm', methods=['GET'])
def lm():
    """Linear model page."""

    return render_template(
        'index_lm.html',
        title='LM main page',
        template='lm-template',
    )
