from flask import Flask


def init_app():
    """Init the main application"""

    app = Flask(__name__, instance_relative_config=False)
    app.config.from_pyfile('config.py')

    with app.app_context():  # structure
        # import routes
        from .home import routes

        # blueprints
        app.register_blueprint(routes.home_bp)

        return app
