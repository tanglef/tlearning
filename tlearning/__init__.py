import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_redis import FlaskRedis


# Globally accessible libraries
db = SQLAlchemy()
r = FlaskRedis()


def init_app():
    """Init the main application"""

    app = Flask(__name__, instance_relative_config=False)
    app.config.from_pyfile('config.py')

    # Initialize Plugins
    db.init_app(app)
    r.init_app(app)

    with app.app_context():  # structure
        # import routes
        from . import routes

        # blueprints
        # app.register_blueprint(auth.auth_bp)
        # app.register_blueprint(admin.admin_bp)

        return app
