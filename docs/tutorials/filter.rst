Filtering
=========

* **Tools -> Filtering**

Instrument noise often causes unwanted spikes in data.
The `Filtering` tool removes spikes by applying a rolling filter across an image.

Available Filters
-----------------

.. table:: Filters implemented in |pewpew|.
    :name: filter_methods
    :align: center

    +----------------+---+--------------------------------------------+
    | Type           | Threshold                                      |
    +================+===+============================================+
    | Rolling Mean   | σ | Distance in stddevs from the local mean.   |
    |                |   | Stddev excludes the tested value.          |
    +----------------+---+--------------------------------------------+
    | Rolling Median | M | Distance in medians from the local median. |
    +----------------+---+--------------------------------------------+

Example: De-noising an image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../images/tutorial_filter_pre.png
    :width: 400px
    :align: center

    Negative spikes (-33) are visible due the to instrument missing an acquisition
    cycle.

1. Using the `Filtering` tool select the appropriate filter.
    Available filters are `Rolling Mean` and `Rolling Median`.

2. Select appropriate filter size.
    Typical window sizes are 3, 5 or 7. A larger window size will be more
    sensitive in detecting outlying data but is more likely to introduce artifacts.

3. Select the appropriate filter threshold.
    An outlying data (those above the thresholds in :ref:`Available Filters`)
    are set to the relevant local value.
    An ideal threshold will change invalid data while leaving valid data untouched.

.. figure:: ../images/tutorial_filter_post.png
    :width: 400px
    :align: center

    A rolling median filter replaces the invalid values with the local median.
