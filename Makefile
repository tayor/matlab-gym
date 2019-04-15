upload:
	make clean
	python3 setup.py sdist bdist_wheel && twine upload dist/*
clean:
	python setup.py clean --all
	pyclean .
	rm -rf *.pyc __pycache__ build dist gym_matlab.egg-info gym_matlab/__pycache__ gym_matlab/units/__pycache__ tests/__pycache__ tests/reports docs/build
