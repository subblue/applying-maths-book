

# Applying Maths in the Chemical and Biomolecular Sciences.

This book is a much updated and corrected version of 'Applying Maths in the Chemical and Biomolecular Sciences, an example based approach' published by OUP 2009. The book is aimed at final year undergraduates and starting graduate students. The aim is to provide examples of the each of the topics that a student would need by giving worked examples with the potential for live computer code if needed.

The code and examples are now all in Python 3 and using NumPy, SciPy and SymPy as necessary for calculations and Matplotlib for graphics. The updated text was written using Jupyter Notebooks. All these packages are free to use, a quite remarkable gift to anyone interested in science, engineering and mathematics. The code for many calculations is included in the text and can be copied or used directly via the 'rocket' icon at the top of any page containing code. The code is written in a simple form, not necessarily the most efficient, and is aimed at gaining an understanding of the calculation.

All the figures are redrawn and many more added. Extra examples have been added to most chapters and some new topics included such as how the fourier transform infra red spectrometer (FTIR) works, how lateral flow and the polymerase chain reaction (PCR) works, the shape of eluted peaks in chromatography, the transient grating technique, molecular beam scattering experiments, the Hilbert transform and Analytic signal, Fourier methods to solve differential equations, 2D Fourier transforms, x-ray diffraction and computed tomography and the Radon transform. The fully worked out solutions to the problems are included at the end of each chapter. 

Godfrey Beddard is Emeritus Professor of Chemistry at the University of Leeds, and Visiting Professor of Chemistry at the University of Edinburgh. He has taught Chemistry for over 30 years, and his research interests are in femtosecond spectroscopy and time-resolved, x-ray crystallography particularly using the Hadamard Transform.

## Preface to original edition

This textbook is primarily intended for final year undergraduate and postgraduate students, although the more elementary parts of the subject are included so as to make the book complete in itself. It is not written with any particular science degree in mind although the emphasis is towards the chemistry and physics of molecules with examples ranging from hydrogen to proteins and DNA.

## Acknowledgements

I would like to thank Dr. David Salt RIP for reading and critically commenting on early drafts of several chapters, David Fogarty for the experimental data used in Chapter 13,  Marcelo de Miranda, Gavin Reid and Briony Yorke for their constructive comments on numerous topics, and to Tom Beddard for help with compiling this web book. I also thank the authors of the books and papers which have formed the basis of several questions and diagrams all of whom I have tried to acknowledge in the text. Finally I thank my family who have received much less of my time and attention than is their due and to whom this work is dedicated.

Contact me via: <a href="&#109;&#97;&#105;&#108;&#116;&#111;&#58;&#103;&#111;&#100;&#102;&#114;&#101;&#121;&#64;&#115;&#117;&#98;&#98;&#108;&#117;&#101;&#46;&#99;&#111;&#109;">&#103;&#111;&#100;&#102;&#114;&#101;&#121;&#64;&#115;&#117;&#98;&#98;&#108;&#117;&#101;&#46;&#99;&#111;&#109;</a>

## Licence 

<a rel="license" href="https://eur03.safelinks.protection.outlook.com/?url=http%3A%2F%2Fcreativecommons.org%2Flicenses%2Fby-nd%2F3.0%2F&amp;data=05%7C01%7CG.S.Beddard%40leeds.ac.uk%7Cce185818f74a491cf66b08daac6ae9f7%7Cbdeaeda8c81d45ce863e5232a535b7cb%7C1%7C0%7C638011872572578072%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&amp;sdata=eZGtZPkaspb1wnZAz89SzuxrExpc3dfIDWu8DN%2BTaGw%3D&amp;reserved=0"><img alt="Creative Commons License" style="border-width:0" src="https://eur03.safelinks.protection.outlook.com/?url=https%3A%2F%2Fi.creativecommons.org%2Fl%2Fby-nd%2F3.0%2F88x31.png&amp;data=05%7C01%7CG.S.Beddard%40leeds.ac.uk%7Cce185818f74a491cf66b08daac6ae9f7%7Cbdeaeda8c81d45ce863e5232a535b7cb%7C1%7C0%7C638011872572578072%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&amp;sdata=geEzVQk994lIjTPodr8VDtGMnjLBH7vkZinGFOKED0E%3D&amp;reserved=0" /></a><br />This work is licensed under a <a rel="license" href="https://eur03.safelinks.protection.outlook.com/?url=http%3A%2F%2Fcreativecommons.org%2Flicenses%2Fby-nd%2F3.0%2F&amp;data=05%7C01%7CG.S.Beddard%40leeds.ac.uk%7Cce185818f74a491cf66b08daac6ae9f7%7Cbdeaeda8c81d45ce863e5232a535b7cb%7C1%7C0%7C638011872572578072%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&amp;sdata=eZGtZPkaspb1wnZAz89SzuxrExpc3dfIDWu8DN%2BTaGw%3D&amp;reserved=0">Creative Commons Attribution-NoDerivs 3.0 Unported License</a>.

10v24.

The \tag{} instruction used to label equations does not work with matrices or when continuation lines are present in the Mathjax. Consequently spaces are used which results in variable positions for equation numbers.