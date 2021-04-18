.PHONY : tests clean install docs

requirements : requirements.txt test-requirements.txt

requirements.txt test-requirements.txt : %.txt : %.in setup.py
	pip-compile --upgrade -v $< --output-file $@

sync : requirements.txt
	pip-sync $<

clean : clean/tests clean/docs
	rm -rf build
	rm -f **/*.so

clean/tests :
	rm -rf .pytest_cache

tests :
	py.test tests --lf --cov=macrodynamics --cov-report=html --cov-report=term-missing \
		--cov-fail-under=100

clean/docs :
	rm -rf docs/build

docs :
	sphinx-build docs docs/build

docs/api :
	sphinx-apidoc --ext-intersphinx --ext-mathjax -o docs macrodynamics
