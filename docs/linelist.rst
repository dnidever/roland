*********
Linelists
*********


Using Linelists in Roland
=========================

|roland| has an :class:`.Abund` class for single abundance values and an :class:`.Abundances` class that holds the abundances for
all elements.  These are handy for modifying abundances when running synthesis.

Let's import the  :class:`.Abundances` class from the :mod:`.abundance` module::

    >>> from roland.abundance import Abundances

Internal Values and Hydrogen/Helium Abundances
----------------------------------------------
    
Internally, an :class:`.Abundances` object stores the abundances values in linear `N(X)/N(H)` format.

The Hydrogen and Helium abundances are special and are held constant at `N(H)/N(H)=1.0` and `N(He)/N(H)=0.0851138`.


Solar Abundance Values
----------------------

The :class:`.Abundances` object stored internal values for solar abundances.  The type of values can be
selected at initialization time.  The options are:

 - "asplund: Asplund, Grevesse and Sauval (2005)
 - "husser": Asplund et al. 2009 chosen for the Husser et al. (2013) Phoenix model atmospheres
 - "kurucz": abundances for the Kurucz model atmospheres (Grevesse+Sauval 1998)
 - "marcs": abundances for the MARCS model atmospheres (Grevesse+2007 values with CNO abundances from Grevesse+Sauval 2008)

The "asplund" values are used by default.
   

Initializing
------------

You can initialize an :class:`.Abundances` object using 1) a list of abundances, 2) a dictionary of abundance values, 3) a single
metallicity or 4) just with the default solar values (no input).

This will use the default solar values::

    >>> abu = Abundances()
    >>> abu
    Abundances([M/H]=0.00,solar=asplund)
    Num Name   N(X)/N(H)   log(eps)   [X/H]
     1   H   1.000000e+00   12.000  [ ---- ]
     2   He  8.511380e-02   10.930  [ ---- ]
     3   Li  1.122018e-11    1.050  [ 0.000]
     4   Be  2.398833e-11    1.380  [ 0.000]
     5   B   5.011872e-10    2.700  [ 0.000]
     6   C   2.454709e-04    8.390  [ 0.000]
     ...
    96   Cm  1.023293e-22   -9.990  [ 0.000]
    97   Bk  1.023293e-22   -9.990  [ 0.000]
    98   Cf  1.023293e-22   -9.990  [ 0.000]
    99   Es  1.023293e-22   -9.990  [ 0.000]
     
Let's start with `[M/H]=-1.0` instead this time:

    >>> abu = Abundances(-1.0)
    >>> abu
    Abundances([M/H]=-1.00,solar=asplund)
    Num Name   N(X)/N(H)   log(eps)   [X/H]
     1   H   1.000000e+00   12.000  [ ---- ]
     2   He  8.511380e-02   10.930  [ ---- ]
     3   Li  1.122018e-12    0.050  [-1.000]
     4   Be  2.398833e-12    0.380  [-1.000]
     5   B   5.011872e-11    1.700  [-1.000]
     6   C   2.454709e-05    7.390  [-1.000]
     ...
    96   Cm  1.023293e-23  -10.990  [-1.000]
    97   Bk  1.023293e-23  -10.990  [-1.000]
    98   Cf  1.023293e-23  -10.990  [-1.000]
    99   Es  1.023293e-23  -10.990  [-1.000]

Now, let's give it a dictionary of abundances values:

    >>> abu = Abundances({"MG_H":-0.5,"SI_H":0.24})
    >>> abu
    Abundances([M/H]=0.00,solar=asplund)
    Num Name   N(X)/N(H)   log(eps)   [X/H]
     1   H   1.000000e+00   12.000  [ ---- ]
     2   He  8.511380e-02   10.930  [ ---- ]
     3   Li  1.122018e-11    1.050  [ 0.000]
     4   Be  2.398833e-11    1.380  [ 0.000]
     5   B   5.011872e-10    2.700  [ 0.000]
     6   C   2.454709e-04    8.390  [ 0.000]
     7   N   6.025596e-05    7.780  [ 0.000]
     8   O   4.570882e-04    8.660  [ 0.000]
     9   F   3.630781e-08    4.560  [ 0.000]
    10   Ne  6.918310e-05    7.840  [ 0.000]
    11   Na  1.479108e-06    6.170  [ 0.000]
    12   Mg  1.071519e-05    7.030  [-0.500]
    13   Al  2.344229e-06    6.370  [ 0.000]
    14   Si  5.623413e-05    7.750  [ 0.240]
    ...
    
Finally, we can give an entire array or list of abundances values.  You have give the type of abundance
values you are giving in the second parameter.  The options are `linear`, `log`, `logeps`, or `x_h`::
    >>> abu = Abundances([12.  , 10.93,  1.05,  1.38,  2.7 ,  8.39,  7.78,
                          8.66,  4.56,  7.84,  6.17,  7.03,  6.37,  7.75,
			  5.36,  7.14,  5.5 ,  6.18],'logeps')
    >>> abu
    Abundances([M/H]=0.00,solar=asplund)
    Num Name   N(X)/N(H)   log(eps)   [X/H]
     1   H   1.000000e+00   12.000  [ ---- ]
     2   He  8.511380e-02   10.930  [ ---- ]
     3   Li  1.122018e-11    1.050  [ 0.000]
     4   Be  2.398833e-11    1.380  [ 0.000]
     5   B   5.011872e-10    2.700  [ 0.000]
     6   C   2.454709e-04    8.390  [ 0.000]
    ...
    97   Bk  1.023293e-22   -9.990  [ 0.000]
    98   Cf  1.023293e-22   -9.990  [ 0.000]
    99   Es  1.023293e-22   -9.990  [ 0.000]


Modifying an Abundances Object
------------------------------

You can always modify an :class:`.Abundances` object `in place`::

    >>> abu['O_H'] = -0.5
    Abundances([M/H]=-0.17,solar=asplund)
    Num Name   N(X)/N(H)   log(eps)   [X/H]
     1   H   1.000000e+00   12.000  [ ---- ]
     2   He  8.511380e-02   10.930  [ ---- ]
     3   Li  1.122018e-11    1.050  [ 0.000]
     4   Be  2.398833e-11    1.380  [ 0.000]
     5   B   5.011872e-10    2.700  [ 0.000]
     6   C   2.454709e-04    8.390  [ 0.000]
     7   N   6.025596e-05    7.780  [ 0.000]
     8   O   1.445440e-04    8.160  [-0.500]
     9   F   3.630781e-08    4.560  [ 0.000]
    ...
    96   Cm  1.023293e-22   -9.990  [ 0.000]
    97   Bk  1.023293e-22   -9.990  [ 0.000]
    98   Cf  1.023293e-22   -9.990  [ 0.000]
    99   Es  1.023293e-22   -9.990  [ 0.000]

Or change the metallicity::

    >>> abu['M_H'] = -0.5    
    Abundances([M/H]=-0.50,solar=asplund)
    Num Name   N(X)/N(H)   log(eps)   [X/H]
     1   H   1.000000e+00   12.000  [ ---- ]
     2   He  8.511380e-02   10.930  [ ---- ]
     3   Li  3.548134e-12    0.550  [-0.500]
     4   Be  7.585776e-12    0.880  [-0.500]
     5   B   1.584893e-10    2.200  [-0.500]
     6   C   7.762471e-05    7.890  [-0.500]
     7   N   1.905461e-05    7.280  [-0.500]
     8   O   1.445440e-04    8.160  [-0.500]
    ...
    95   Am  3.235937e-23  -10.490  [-0.500]
    96   Cm  3.235937e-23  -10.490  [-0.500]
    97   Bk  3.235937e-23  -10.490  [-0.500]
    98   Cf  3.235937e-23  -10.490  [-0.500]
    99   Es  3.235937e-23  -10.490  [-0.500]

You can also change the entire metallicity by an increment amount::
    
    >>> abu += 0.5
    Abundances([M/H]=0.50,solar=asplund)
    Num Name   N(X)/N(H)   log(eps)   [X/H]
     1   H   1.000000e+00   12.000  [ ---- ]
     2   He  8.511380e-02   10.930  [ ---- ]
     3   Li  3.548134e-11    1.550  [ 0.500]
     4   Be  7.585776e-11    1.880  [ 0.500]
     5   B   1.584893e-09    3.200  [ 0.500]
     6   C   7.762471e-04    8.890  [ 0.500]
     7   N   1.905461e-04    8.280  [ 0.500]
    ...
    96   Cm  3.235937e-22   -9.490  [ 0.500]
    97   Bk  3.235937e-22   -9.490  [ 0.500]
    98   Cf  3.235937e-22   -9.490  [ 0.500]
    99   Es  3.235937e-22   -9.490  [ 0.500]
    
Or the alpha abundances::

    >>> abu['alpha'] -= 0.5
    Abundances([M/H]=-0.25,solar=asplund)
    Num Name   N(X)/N(H)   log(eps)   [X/H]
     1   H   1.000000e+00   12.000  [ ---- ]
     2   He  8.511380e-02   10.930  [ ---- ]
     3   Li  1.122018e-11    1.050  [ 0.000]
     4   Be  2.398833e-11    1.380  [ 0.000]
     5   B   5.011872e-10    2.700  [ 0.000]
     6   C   2.454709e-04    8.390  [ 0.000]
     7   N   6.025596e-05    7.780  [ 0.000]
     8   O   1.445440e-04    8.160  [-0.500]
     9   F   3.630781e-08    4.560  [ 0.000]
    10   Ne  2.187762e-05    7.340  [-0.500]
    11   Na  1.479108e-06    6.170  [ 0.000]
    12   Mg  1.071519e-05    7.030  [-0.500]
    13   Al  2.344229e-06    6.370  [ 0.000]
    14   Si  1.023293e-05    7.010  [-0.500]
    15   P   2.290868e-07    5.360  [ 0.000]
    16   S   4.365158e-06    6.640  [-0.500]
    17   Cl  3.162278e-07    5.500  [ 0.000]
    18   Ar  4.786301e-07    5.680  [-0.500]
    19   K   1.202264e-07    5.080  [ 0.000]
    20   Ca  6.456542e-07    5.810  [-0.500]
    21   Sc  1.122018e-09    3.050  [ 0.000]
    22   Ti  2.511886e-08    4.400  [-0.500]
    23   V   1.000000e-08    4.000  [ 0.000]
    ...
    96   Cm  1.023293e-22   -9.990  [ 0.000]
    97   Bk  1.023293e-22   -9.990  [ 0.000]
    98   Cf  1.023293e-22   -9.990  [ 0.000]
    99   Es  1.023293e-22   -9.990  [ 0.000]
    

Creating a New, Modified Abundances Object
------------------------------------------

You can also `call` the object and create a new, modified object.

Create a new :class:`.Abundances` object with a metallicity of -1.5::

    >>> abu2 = abu(-1.5)
    >>> abu2
    Abundances([M/H]=-1.50,solar=asplund)
    Num Name   N(X)/N(H)   log(eps)   [X/H]
     1   H   1.000000e+00   12.000  [ ---- ]
     2   He  8.511380e-02   10.930  [ ---- ]
     3   Li  3.548134e-13   -0.450  [-1.500]
     4   Be  7.585776e-13   -0.120  [-1.500]
     5   B   1.584893e-11    1.200  [-1.500]
     6   C   7.762471e-06    6.890  [-1.500]
    ...
    96   Cm  3.235937e-24  -11.490  [-1.500]
    97   Bk  3.235937e-24  -11.490  [-1.500]
    98   Cf  3.235937e-24  -11.490  [-1.500]
    99   Es  3.235937e-24  -11.490  [-1.500]
    
You can also input a dictionary of abundances values::

    >>> abu2 = abu({"c_h":-1.5})
    >>> abu2
    Abundances([M/H]=-0.12,solar=asplund)
    Num Name   N(X)/N(H)   log(eps)   [X/H]
     1   H   1.000000e+00   12.000  [ ---- ]
     2   He  8.511380e-02   10.930  [ ---- ]
     3   Li  1.122018e-11    1.050  [ 0.000]
     4   Be  2.398833e-11    1.380  [ 0.000]
     5   B   5.011872e-10    2.700  [ 0.000]
     6   C   7.762471e-06    6.890  [-1.500]
     7   N   6.025596e-05    7.780  [ 0.000]
    ...
    96   Cm  1.023293e-22   -9.990  [ 0.000]
    97   Bk  1.023293e-22   -9.990  [ 0.000]
    98   Cf  1.023293e-22   -9.990  [ 0.000]
    99   Es  1.023293e-22   -9.990  [ 0.000]
    
Abundances Output
-----------------

The :class:`.Abundances` class can output the information in several ways.

If you select a single element (by element name or index), an :class:`.Abund` object will be returned.::

    >>> abu['Ca']
    Abund(20 Ca N(Ca)/N(H)=2.042e-06 log(eps)=6.310)

    >>> abu[10]
    Abund(11 Na N(Na)/N(H)=1.479e-06 log(eps)=6.170)

Selecting values in bracket notation will only return the value.  Abundance versus H::

    >>> abu['Ca_H']
    0.0

Abundance versus M::

    >>> abu['Ca_M']
    -0.0003221051142099841
    
There are several useful properties that will print out **all** of the abundances.

Print the linear or `N(X)/N(H)` values with `linear`.::

    >>> abu.linear
    array([1.00000000e+00, 8.51138038e-02, 1.12201845e-11, 2.39883292e-11,
       5.01187234e-10, 2.45470892e-04, 6.02559586e-05, 4.57088190e-04,
       3.63078055e-08, 6.91830971e-05, 1.47910839e-06, 1.07151931e-05,
       ...
       1.02329299e-22, 1.14815362e-12, 1.02329299e-22, 3.01995172e-13,
       1.02329299e-22, 1.02329299e-22, 1.02329299e-22, 1.02329299e-22,
       1.02329299e-22, 1.02329299e-22, 1.02329299e-22])

Or you can also use `log`, `logeps`, `xh`, or `xm`.  The `log` abundances::
    >>> abu.log
    array([  0.  ,  -1.07, -10.95, -10.62,  -9.3 ,  -3.61,  -4.22,  -3.34,
            -7.44,  -4.16,  -5.83,  -4.97,  -5.63,  -4.25,  -6.64,  -4.86,
            -6.5 ,  -5.82,  -6.92,  -5.69,  -8.95,  -7.1 ,  -8.  ,  -6.36,
	   ...
	   -11.1 , -10.  , -11.35, -21.99, -21.99, -21.99, -21.99, -21.99,
	   -21.99, -11.94, -21.99, -12.52, -21.99, -21.99, -21.99, -21.99,
	   -21.99, -21.99, -21.99])

Abundances in `log(eps)` notation, or `log(N(X)/N(H))+12.0`::
  
    >>> abu.logeps
    array([12.  , 10.93,  1.05,  1.38,  2.7 ,  8.39,  7.78,  8.66,  4.56,
        7.84,  6.17,  7.03,  6.37,  7.75,  5.36,  7.14,  5.5 ,  6.18,
        5.08,  6.31,  3.05,  4.9 ,  4.  ,  5.64,  5.39,  7.45,  4.92,
	...
       -0.17,  1.11,  0.23,  1.45,  1.38,  1.64,  1.01,  1.13,  0.9 ,
        2.  ,  0.65, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99,  0.06,
       -9.99, -0.52, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99])

Bracket notation, relative to H::

    >>> abu.xh
    array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  , -0.5 ,  0.  ,  0.24,  0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
	    ...
            0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ])
		
Bracket notation, relative to M::
  
    >>> abu.xm    
    array([-3.22105114e-04, -3.22105114e-04, -3.22105114e-04, -3.22105114e-04,
           -3.22105114e-04, -3.22105114e-04, -3.22105114e-04, -3.22105114e-04,
           -3.22105114e-04, -3.22105114e-04, -3.22105114e-04, -5.00322105e-01,
	   ...
           -3.22105114e-04, -3.22105114e-04, -3.22105114e-04, -3.22105114e-04,
           -3.22105114e-04, -3.22105114e-04, -3.22105114e-04, -3.22105114e-04,
           -3.22105114e-04, -3.22105114e-04, -3.22105114e-04])

You can also return the abundances in formats that are useful for model atmospheres.

Return abundance values in the format for Kurucz model atmospheres::
  
    >>> abu.to_kurucz()
    array([  0.92075543,   0.07836899, -10.98585571, -10.65585571,
            -9.33585571,  -3.64585571,  -4.25585571,  -3.37585571,
            -7.47585571,  -4.19585571,  -5.86585571,  -4.50585571,
            -6.64585571,  -4.58585571,  -7.11585571,  -5.80585571,
	    ...
            -20.        , -11.97585571, -20.        , -12.55585571,
            -20.        , -20.        , -20.        , -20.        ,
            -20.        , -20.        , -20.        ]

	    
Return abundance values in the format for MARCS model atmospheres::
  
    >>> abu.to_marcs()
    array([12.  , 10.93,  1.05,  1.38,  2.7 ,  8.39,  7.78,  8.66,  4.56,
           7.84,  6.17,  7.53,  6.37,  7.51,  5.36,  7.14,  5.5 ,  6.18,
           5.08,  6.31,  3.05,  4.9 ,  4.  ,  5.64,  5.39,  7.45,  4.92,
	   ...
          -0.17,  1.11,  0.23,  1.45,  1.38,  1.64,  1.01,  1.13,  0.9 ,
           2.  ,  0.65, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99,  0.06,
          -9.99, -0.52, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99])

Other useful properties and methods::

    # Return the metallicity as [M/H]
    >>> abu.metallicity
    0.0

    # Return the metallicity as Sum(N(X)/N(H) over all metals
    >>> abu.metals
    0.0009509329459494126

    # Return all of the element symbols
    >>> abu.symbol
    ['H','He','Li','Be','B','C','N','O','F',
    ...
     'U','Np','Pu','Am','Cm','Bk','Cf','Es']

    # Return all of the element mass values (in amu).
    >>> abu.mass
    [1.00794, 4.0026, 6.941, 9.01218, 10.811, 12.0107,
    ...
     244.0, 243.0, 247.0, 247.0, 251.0, 252.0]
