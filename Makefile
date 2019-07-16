.PHONY : tests clean install

requirements.txt test-requirements.txt : %.txt : %.in setup.py
	pip-compile --upgrade -v $< --output-file $*.tmp
	./make_paths_relative.py < $*.tmp > $@

install :
	pip install -r requirements.txt

clean : clean/tests
	rm -rf build
	rm -f **/*.so

clean/tests :
	rm -rf .pytest_cache

tests :
	py.test tests --lf --cov=graph_dynamics --cov-report=html --cov-report=term-missing
