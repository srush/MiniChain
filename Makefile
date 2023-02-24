INPUTS = $(wildcard *.py) $(wildcard */*.py)

OUTPUTS = $(patsubst %.py,%.ipynb,$(INPUTS))

examples/%.ipynb : examples/%.py
	jupytext --execute --to notebook $<
