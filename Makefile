install:
	python -m pip install --upgrade pip &&\
	python -m pip install -e .[dev]

lint:
	echo 'Not implemented'

test:
	echo 'Not implemented'

format:
	black awb/*.py