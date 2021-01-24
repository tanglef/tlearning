from flask import Flask


def init_app():
    """Init the main application"""

    app = Flask(__name__, instance_relative_config=False)
    app.config.from_pyfile('config.py')

    with app.app_context():  # structure
        # import routes
        from .home import routes as routes_home
        from .lm import routes as routes_lm
        from .logreg import routes as routes_logreg
        # blueprints
        app.register_blueprint(routes_home.home_bp)
        app.register_blueprint(routes_lm.lm_bp)
        app.register_blueprint(routes_logreg.logreg_bp)

        return app
