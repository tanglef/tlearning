from setuptools import find_packages
from setuptools import setup

setup(
    name='tlearning',
    description="Machine learning introduction app",
    author='Tanguy Lefort',
    url='https://tlearning.herokuapp.com/',
    packages=find_packages('tlearning'),
    package_dir={
        '': 'tlearning'},
    include_package_data=True,
    keywords=[
        'app', 'test', 'flask', 'machine_learning'
    ],
    entry_points={
        'console_scripts': [
            'web_server = __init__:init_app']},
)
