install:
	pip install --upgrade pip &&\
		conda env create -f environment.yml
		pip install fastai==2.7.9

# format:
#	black *.py

#lint:
#	pylint --disable=R,C app.py

# test:
#	#python -m pytest -vv test_flask_app.py

all: install format lint test
