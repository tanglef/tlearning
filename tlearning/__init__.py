from flask import Flask
from .uncertainty_apps import character_stratagy, cifar10h

dash_chara = character_stratagy.dash_application()
# dash_cifar = cifar10h.dash_application()


def init_app():
    """Init the main application"""

    app = Flask(__name__, instance_relative_config=False)
    app.config.from_pyfile('config.py')

    dash_chara.init_app(app=app)
    # dash_cifar.init_app(app=app)

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
