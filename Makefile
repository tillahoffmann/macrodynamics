.PHONY : tests clean install

requirements.txt test-requirements.txt : %.txt : %.in setup.py
	pip-compile -v $< --output-file $*.tmp
	./make_paths_relative.py < $*.tmp > $@

install :
	pip install -r requirements.txt

clean :
	rm -rf build
	rm -f **/*.so

tests :
	py.test tests --cov=graph_dynamics --cov-report=html --cov-report=term-missing
