install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# format:
#	black *.py

#lint:
#	pylint --disable=R,C app.py

# test:
#	#python -m pytest -vv test_flask_app.py

all: install format lint test
