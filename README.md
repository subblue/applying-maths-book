# Applying Maths in the Chemical and Biomolecular Sciences

This book is a much updated and corrected version of &lsquo;[Applying Maths in the Chemical and Biomolecular Sciences, an example based approach](https://www.amazon.co.uk/Applying-Maths-Chemical-Biomolecular-Sciences/dp/0199230919)&rsquo; published by OUP 2009. 

The code and examples are now all in Python 3 and using numpy, scipy and sympy as necessary for calculations and matplotlib for graphics. The updated text was written using Jupyter notebooks. 

All these packages are free to use, a quite remarkable gift to anyone interested in science, engineering and mathematics. The code for many calculations is included in the text and can be copied and used diectly into a notebook mostly with only minor changes. The code is written in a simple form, not necessarily the most efficient, and is aimed at gaining an undersatanding of the calculation.

More examples have been added to most chapters, for example in chemical kinetics and some new topics added such as fourier methods to solve differential equations, x-ray diffraction and computed tomography. The fully worked out solutions to the problems are included at the end of each chapter.

***Godfrey Beddard Feb. 2022.***



## Building the book

If you'd like to develop and/or build the Applying Maths Book book, you should:

1. Clone this repository
2. Run `pip install -r requirements.txt` (it is recommended you do this within a virtual environment)
3. (Optional) Edit the books source files located in the `applying_maths_book/` directory
4. Run `jupyter-book clean applying_maths_book/` to remove any existing builds
5. Run `jupyter-book build applying_maths_book/`

A fully-rendered HTML version of the book will be built in `applying_maths_book/_build/html/`.

### Credits

This project is created using the excellent open source [Jupyter Book project](https://jupyterbook.org/) and the [executablebooks/cookiecutter-jupyter-book template](https://github.com/executablebooks/cookiecutter-jupyter-book).

### License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
