"""
===================
Test the flask app
===================
"""

import pytest
import sys
import os
import glob

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tlearning import init_app  # noqa


@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # create the app with common test config
    app = init_app()
    yield app


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_base_home(client):
    """Test the base template for home."""

    rv = client.get('/')
    assert rv.data[0:15] == b"<!doctype html>"


def test_lm(client):
    """Test the plotly graphs for lm are created."""

    files = glob.glob(os.path.join(parentdir, "tlearning", "lm",
                                   "static", "*.html"))
    files = [os.path.basename(f) for f in files]
    assert 'linear_trend.html' in files
    assert 'exp_trend.html' in files
