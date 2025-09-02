.PHONY : tests clean install docs

clean : clean/tests clean/docs

clean/tests :
	rm -rf .pytest_cache

tests :
	py.test tests -v --ff --cov=macrodynamics --cov-report=html \
		--cov-report=term-missing --cov-fail-under=100

lint :
	ruff format --check .

clean/docs :
	rm -rf docs/_build

docs :
	sphinx-build . docs/_build
