.PHONY : tests clean install docs

requirements.txt : requirements.in setup.py
	pip-compile --upgrade -v $< --output-file $@

sync : requirements.txt
	pip-sync $<

clean : clean/tests clean/docs
	rm -rf build
	rm -f **/*.so

clean/tests :
	rm -rf .pytest_cache

tests :
	py.test tests -v --ff --cov=macrodynamics --cov-report=html --cov-report=term-missing \
		--cov-fail-under=100

clean/docs :
	rm -rf docs/_build

docs :
	sphinx-build . docs/_build
