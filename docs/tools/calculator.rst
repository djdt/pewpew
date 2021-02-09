Calculator
==========

.. meta::
   :keywords: calculator, calculation, parser, formula

* **Tools -> Calculator**

The built in `Calculator` can perform simple calculations on image data by
entering the desired formula into the `Formula` text box.
For types of supported input and examples see :ref:`calc_input` and for
supported functions see :ref:`calc_functions`.
These functions are also available for insert via the `Functions` combo box.

Clicking apply will create a new image with the name given in `Name`,
if the name already exists then the image data will be overwritten.
Non image results (e.g. the mean of an image) are displayed in the `Results` text box.

Supported Input
---------------

.. table:: Supported `Formula` inputs with examples.
    :name: calc_input
    :align: center

    +--------------------+------------------------------------------------+
    |Type                |Example                                         |
    +====================+================================================+
    |Image names         |``P31``                                         |
    |                    |                                                |
    |                    |``153Eu/31P``                                   |
    +--------------------+------------------------------------------------+
    |Mathematical Symbols|``+  -  *  /  ^  (  )``                         |
    |                    |                                                |
    |                    |``a / (b + 1)``                                 |
    |                    |                                                |
    |                    |``1e3 + P31``                                   |
    +--------------------+------------------------------------------------+
    |If / then / else    |``if P31 > 1e3 then P31 else 0``                |
    |                    |                                                |
    |                    |``Eu153 > P31 ? Eu153 : P31``                   |
    +--------------------+------------------------------------------------+
    |Functions           |``threshold(P31, median(P31))``                 |
    |                    |                                                |
    |                    |``if P31 > kmeans(P31, 3)[2] then P31 else nan``|
    +--------------------+------------------------------------------------+

Supported Formulas
------------------

.. table:: Calculator functions.
    :name: calc_functions
    :align: center

    +----------+-----------------+------------------------------------+
    |Function  |Arguments        |Result                              |
    +==========+=================+====================================+
    |abs       |image            |absolute values of image            |
    +----------+-----------------+------------------------------------+
    |kmeans    |image, k         |array of lower k-means bounds       |
    +----------+-----------------+------------------------------------+
    |mean      |image            |mean value of the image             |
    +----------+-----------------+------------------------------------+
    |median    |image            |median value of the image           |
    +----------+-----------------+------------------------------------+
    |nantonum  |image            |sets image NaN values to zero       |
    +----------+-----------------+------------------------------------+
    |normalise |image, min, max  |image normalised to `min`, `max`    |
    +----------+-----------------+------------------------------------+
    |otsu      |image            |Otsu's method of image              |
    +----------+-----------------+------------------------------------+
    |percentile|image, percentile|the `percentile`'th value of image  |
    +----------+-----------------+------------------------------------+
    |threshold |image, value     |sets all pxiels below `value` to nan|
    +----------+-----------------+------------------------------------+
