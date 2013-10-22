all:
	python setup.py develop --user

doc:
	python setup.py build_sphinx

clean:
	rm -rf build

.PHONY: all doc clean
