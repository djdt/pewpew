.. pew documentation master file, created by
   sphinx-quickstart on Thu Nov 19 12:53:38 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|pewpew|
========

|pewpew| is an open-source LA-ICP-MS data import and processing application, based on the python library
pewlib_.
Currently we support the following formats via drag-and-drop and the :ref:`Import Wizard`:

* Agilent MassHunter batches (.b)
* PerkinElmer XL (.xl)
* Thermo Qtega (.csv and LDR)
* Nu Instruments (.csv exports)
* TOFWERKs (.csv exports)
* izmML (.imzML and .idb), see :ref:`imzML Import Wizard`

Support for import and alignment of data with NWL laser logs is avaible via the :ref:`LaserLog Import Wizard`.

To get started :ref:`Install<Installation>` the program then check the :ref:`Basic Usage`.

.. toctree::
    :hidden:
    :maxdepth: 1

    install
    usage

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Dialogs and Wizards

    dialogs/import
    dialogs/calibration
    dialogs/colocal
    dialogs/stats
    dialogs/selection

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Tools

    tools/calculator
    tools/filter
    tools/standards

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Examples
    :numbered:

    examples/import
    examples/calculator
    examples/filter
    examples/standards
    examples/stats
    examples/overlay


.. _pewlib: https://github.com/djdt/pewlib

.. Indices and tables
.. ==================
..
.. * :ref:`genindex`
..
.. * :ref:`search`

.. * :ref:`modindex`
