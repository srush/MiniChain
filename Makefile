INPUTS = $(wildcard examples/*.py)

OUTPUTS = $(patsubst %.py,%.ipynb,$(INPUTS))

examples/%.ipynb : examples/%.py
	python examples/process.py < $< > /tmp/out.py
	jupytext --to notebook /tmp/out.py -o $@

examples/%.md : examples/%.py
	jupytext --to markdown $<

all: $(OUTPUTS)
