.PHONY : tests clean install docs

requirements.txt test-requirements.txt : %.txt : %.in setup.py
	pip-compile --upgrade -v $< --output-file $*.tmp
	./make_paths_relative.py < $*.tmp > $@

install :
	pip install -r requirements.txt

clean : clean/tests clean/docs
	rm -rf build
	rm -f **/*.so

clean/tests :
	rm -rf .pytest_cache

tests :
	py.test tests --lf --cov=graph_dynamics --cov-report=html --cov-report=term-missing \
		--cov-fail-under=100

clean/docs :
	rm -rf docs/build

docs :
	sphinx-build docs docs/build

docs/api :
	sphinx-apidoc --ext-intersphinx --ext-mathjax -o docs graph_dynamics
