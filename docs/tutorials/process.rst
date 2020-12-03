Image Processing
================

Calculator
----------

* **Tools -> Edit Tool -> Calculator**

The built in `Calculator` can perform simple calculations on image data by
entering the desired formula into the `Formula` text box.
The formula box will turn red if the input is invalid.
For types of supported input and examples see :ref:`calc_input` and for
supported functions see :ref:`calc_functions`.
The `Results` text box will show numerical results.

Supported Input
~~~~~~~~~~~~~~~

.. table:: Supported `Formula` inputs with examples.
    :name: calc_input

    +--------------------+------------------------------------------------+
    |Type                |Example                                         |
    +====================+================================================+
    |Image names         |``P31``                                         |
    |                    |                                                |
    |                    |``153Eu/31P``                                   |
    +--------------------+------------------------------------------------+
    |Mathematical Symbols|``+  -  \*  /  ^  (  )``                        |
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
~~~~~~~~~~~~~~~~~~

.. table:: Calculator functions.
    :name: calc_functions

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


Example: Thresholded division of elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Division of two elements is commonly used to normalise one element to another.
Without thresholding of low background values this can produce a rather unattractive image.

.. centered::
    |calc_img_bad| = |calc_img_z66| / |calc_img_p31|

In this example we use the calculator tool to simultaneously threshold and divide two elements.

.. centered::
    |calc_img_div| = |calc_img_z66| / |calc_img_p31|

.. |calc_img_z66| image:: ../images/tutorial_calc_Zn66.png
    :width: 150px
.. |calc_img_p31| image:: ../images/tutorial_calc_p31.png
    :width: 150px
.. |calc_img_div| image:: ../images/tutorial_calc_ZndivP.png
    :width: 150px
.. |calc_img_bad| image:: ../images/tutorial_calc_baddiv.png
    :width: 150px


1. Determine the background value of the divisor image.
    In this example a value of 100 sufficed.

2. Using the calculator tool perform the operation.
    Enter ``if P31 > 100 then Zn66 / P31 else 0`` into the `Formula` box.

    The first part of the if/then/else masks the data so only values above the threshold are
    operated on. The second part performs the division while the last part sets unmasked data to 0.

3. Enter the new name, and click `Apply`.
    Choose a unique name to prevent overwriting data.


Convolution
-----------

* **Tools -> Edit Tool -> Convolve / Deconvolve**

The `Convolve` and `Deconvolve` tools
Convolution can be used to add blur to images.


Example: Removing wash-out blur
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../images/tutorial_convolve_pre.png
    :width: 300px
    :align: center

    An image with wash-out blur, seen on the right hand side of the tissue.

1. Calculate the wash-out time in pixels.
    If the wash-out time is known then use wash-out (s) / acquisition time (s).

2. `Deconvolve` in one-dimension with a non-symmetrical point-spread-function.
    In this example `Log-Normal` (size=12; σ=1.50, μ=0.0) worked well.

3. Optional, to remove deconvolution artefacts take the absolute value of the image.
    Use the abs() function in the `Calculator`.


.. figure:: ../images/tutorial_convolve_post.png
    :width: 300px
    :align: center

    The same image post-deconvolution. Notice the lessen blur on the right hand side.


Filtering
=========

* **Tools -> Edit Tool -> Filter**

Instrument noise often causes unwanted spikes in data.
The `Filter` tool removes spikes by applying a rolling filter across an image.

Example: De-noising an image
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../images/tutorial_filter_pre.png
    :width: 300px
    :align: center

    Negative spikes (-33) are visible due the to instrument missing an acquisition
    cycle.

1. Using the `Filter` tool select the appropriate filter.
   Available filters are `Mean` and `Median`.

2. Select appropriate filter size.
   Typical window sizes are 3, 5 or 7. A larger window size will be more
   sensitive in detecting outlying data but is more likely to introduce artifacts.

3. Select the appropriate filter threshold.
   For `Median` the `M` value is the distance (in medians) a value must be from the
   local median to be considered an outlier.
   For `Mean` the `σ` value is the distance (in standard deviations, excluding the tested point)
   a value must be from the local mean to be considered an outlier.
   An ideal threshold will change invalid data while leaving valid data untouched.

.. figure:: ../images/tutorial_filter_post.png
    :width: 300px
    :align: center

    A rolling mean filter replaces the invalid values with the local mean.
