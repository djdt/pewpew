Drift Compensation
==================

* **Tools -> Drift Compensation**

Drift in signal can be caused by changes in laser, plasma or mass-spec conditions and
can be minimised by using short runs and correctly 'warming up' the ICP-MS.
In cases where quantification is not required then drift can be compensated using the
`Drift Compensation` tool.
This tool fits a polynomial to a section of the image and then normalises with
respect to the fit.


Example: Compensating Laser Drift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../images/tutorial_filter_pre.png
    :width: 300px
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
    :width: 300px
    :align: center

    A rolling mean filter replaces the invalid values with the local mean.
