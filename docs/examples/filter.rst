Example: De-noising an image
============================

In this example we remove instrument noise from an image.
See also :ref:`Filtering Tool`.

.. figure:: ../images/tutorial_filter_pre.jpg
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
    An ideal threshold will change invalid data while leaving valid data untouched.

.. figure:: ../images/tutorial_filter_post.jpg
    :width: 400px
    :align: center

    A rolling median filter replaces the invalid values with the local median.
