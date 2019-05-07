TEST_PATH=./
MYPY_PATH=./

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
lint:
	flake8 --ignore=E501,F401,E128,E402,E731,F821,E712,W503
test: clean
	python3 -m pytest --verbose --color=yes $(TEST_PATH)
mypy:
	python3 -m mypy $(MYPY_PATH) --ignore-missing-imports
activate:
	pipenv shell
install:
	pipenv install