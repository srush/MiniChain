examples/bash.ipynb: examples/bash.py
	jupytext --to ipynb --execute examples/bash.py 

examples/bash.html: examples/bash.ipynb
	jupyter nbconvert --to html examples/bash.ipynb

examples/selfask.ipynb: examples/selfask.py
	jupytext --to ipynb --execute examples/selfask.py 

examples/selfask.html: examples/selfask.ipynb
	jupyter nbconvert --to html examples/selfask.ipynb
