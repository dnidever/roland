.. synthpy documentation master file, created by
   sphinx-quickstart on Tue Feb 16 13:03:42 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*******
SynthPy
*******

Introduction
============
|synthpy| is a generic stellar spectral synthesis package that can be run from python.

.. toctree::
   :maxdepth: 1

   install
   linelist
   atmosphere
   abundance
   spectrumizer
   modules
	      

Description
===========
To run |synthpy| you need 1) a model atmosphere, 2) a linelist (or multiple), and 3) the set of stellar parameters
and elemental abundances that you want to run.

1) Model Atmospheres

   MOOG can read Kurucz/ATLAS or MARCS model atmospheres.  See pages 12-17 in the `MOOG manual <_static/WRITEnov2019.pdf>`_ for the format and examples.

2) Linelists

   MOOG requires a specific linelist format.  See pages 20-21 in the `MOOG manual <_static/WRITEnov2019.pdf>`_ for the format.
   
3) Stellar parameters and elemental abundances.

   The main stellar parameters are Teff, logg, [M/H], and [alpha/M].  These are the first four parameters in the
   main ``synthesis.synthesize()`` function.  The individual elements abundances are given in the ``elems`` parameters
   as a list of [element name, abundance] pairs, where abundance in the in [X/M] format relative to the overall metallicity.
   

Examples
========

.. toctree::
    :maxdepth: 1

    examples

*****
Index
*****

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
