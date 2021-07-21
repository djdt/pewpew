Installation
============
.. index:: Installation

Executables for major |pewpew| releases are available on the `github <https://github.com/djdt/pewpew/releases>`_.

Via Pip
-------

To install |pewpew| as a local python package first clone the repository.
This requires you have C build tools installed on your system.

.. code-block:: bash

    $ git clone https://github.com/djdt/pewpew

Then enter the newly created directory.

.. code-block:: bash

    $ cd pewpew

And install via pip.

.. code-block:: bash

    $ pip install -r requirements.txt
    $ pip install -e .

You can then run |pewpew| from your terminal emulator using ``pewpew`` or as a module from the |pewpew| root directory.

.. code-block:: bash

   $ cd pewpew
   $ python -m pewpew
