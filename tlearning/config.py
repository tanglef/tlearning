"""Flask configuration."""

from os import environ, path
from dotenv import load_dotenv

basedir = path.dirname(path.abspath(path.dirname(__file__)))
print(basedir)
load_dotenv(path.join(basedir, "venv", '.env'))

TESTING = True
DEBUG = True
STATIC_FOLDER = 'static'
TEMPLATES_FOLDER = 'templates'
FLASK_ENV = 'development'
SECRET_KEY = environ.get('SECRET_KEY')
