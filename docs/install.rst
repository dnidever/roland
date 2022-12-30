************
Installation
************


Important packages
==================
|synthpy| is a package to run the `MOOG <https://github.com/jsobeck/MOOG-SCAT_basic_git>`_ and `moog17scat <https://github.com/alexji/moog17scat>`_
spectral synthesis code by Chris Sneden and scattering improvements by Jennifer Sobeck and other updates by Alex Ji.
There is also a Python wrapper/driver based on some old IDL software I wrote and some code from Jon Holtzman in the
in the `APOGEE package <https://github.com/sdss/apogee>`_).

Installing SynthPy
==================

The easiest way to install the code is with pip.

.. code-block:: bash

    pip install synthpy

Synthesis Packages
==================

The various spectral synthesis packages that |synthpy| calls are not installed automatically.  That's because the code
is mostly Fortran or Julia and I didn't trust an automatic install for all of these.  However, they should be pretty
to install.

1) Synspec
----------

``Synspec`` is a package I created to redistribute the `Synspec <http://tlusty.oca.eu/Synspec49/synspec.html>`_ spectral
synthesis package by Ivan Hubeny and Thierry Lanz and Python driver software mostly from Carlos Allende Prieto's
`synple <https://github.com/callendeprieto/synple>`_ package.

The easiest way to install the code is with pip.  This will compile the Fortran code and install the binaries and Python software.

.. code-block:: bash

    pip install synspec


2) Moogpy
---------

`Moogpy <https://github.com/dnidever/moogpy>`_ is a package I created to redistribute the `MOOG <https://www.as.utexas.edu/~chris/moog.html>`_
spectral synthesis package by Chris Sneden with scattering improvements by `Jennifer Sobeck <https://github.com/jsobeck/MOOG-SCAT_basic_git>`_
and other updates by `Alex Ji <https://github.com/alexji/moog17scat>`_.  Python driver software is also included.

The easiest way to install the code is with pip.  This will compile the Fortran code and install the binaries and Python software.

.. code-block:: bash

    pip install moogpy


3) Turbospectrum
----------------

`Turbospectrum <https://github.com/dnidever/turbospectrum>`_ is a package I created to redistribute the
`Turbospectrum <https://github.com/bertrandplez/Turbospectrum_NLTE>`_ spectral synthesis package by Bertrand Plez
and Python driver software.  

The easiest way to install the code is with pip.  This will compile the Fortran code and install the binaries and Python software.

.. code-block:: bash

    pip install turbospectrum


4) Korg
-------

`Korg <https://github.com/ajwheeler/Korg.jl>`_ is a new spectral synthesis package written by `Adam Wheeler <https://arxiv.org/abs/2211.00029>`_
in the Julia programming language.

``Korg`` can be called from Python.  You'll need to install ``Julia`` on your system and install the ``IJulia``, ``PyPlot`` and ``Korg``.

**Installing Julia**

There are ``Julia`` installers for various systems available from the `downloads <https://julialang.org/downloads/>`_ page.
Once I installed ``Julia`` on my Mac laptop, I had to run a few commands to make it callable:

.. code-block:: bash

    sudo mkdir -p /usr/local/bin
    sudo rm -f /usr/local/bin/julia
    sudo ln -s /Applications/Julia-1.8.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia

**Installing Julia packages**

After you have installed ``Julia`` it's pretty straightforward to download and install package directly from ``Julia`` itself.
It has its own built-in package manager.

Start up Julia from the command-line:

.. code-block:: julia

    % julia
               _
       _       _ _(_)_     |  Documentation: https://docs.julialang.org
      (_)     | (_) (_)    |
       _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
      | | | | | | |/ _` |  |
      | | |_| | | | (_| |  |  Version 1.8.3 (2022-11-14)
     _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
    |__/                   |

    julia> 

To download and install a Julia type a right bracket "]", this will take you into the Pkg package manager.

.. code-block:: julia
		
    julia> ]

Then you use the ``add`` command to install packages.  We need to add "Korg", "IJulia", "PyPlot", and "PyCall"

.. code-block::	julia

    (@v1.8) pkg> add Korg
    ... messages
    (@v1.8) pkg>

You can see the packages that you have installed with the ``status`` command.

.. code-block::	julia
    
    (@v1.8) pkg> status
    Status `~/.julia/environments/v1.8/Project.toml`
      [7073ff75] IJulia v1.23.3
      [acafc109] Korg v0.12.1
      [438e738f] PyCall v1.94.1
      [d330b81b] PyPlot v2.11.0
    

**Installing PyJulia**

We also need to install ``pyjulia`` which is a Python package to communicate directly with Julia.
It should be straightforward to pip install it.

.. code-block:: bash

    pip install pyjulia
    

Dependencies
============

- numpy
- scipy
- astropy
- matplotlib
- `dlnpyutils <https://github.com/dnidever/dlnpyutils>`_
