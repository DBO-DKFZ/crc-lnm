from setuptools import setup

setup(
    name="hip",
    version="0.1.0",
    packages=["hip"],
    scripts=["scripts/tile_train.py", "scripts/tile_test.py"],
    package_data={"hip": ["cfg"]},
)
