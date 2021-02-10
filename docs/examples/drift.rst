Example: Compensating laser drift
=================================

In this example we compensate for changes in laser power during ablation of a sample.
See also :ref:`Drift Compensation Tool`.

.. figure:: ../images/tutorial_drift_pre.png
    :width: 400px
    :align: center

    An image with signal drift.

1. Select drift area.
    Drag the white guides to cover an area where no sample is present,
    such as the start or end of an image.
    If there sample is present throughout an image then use the trim controls
    (`Show drift trim controls.`) to remove the sample area.

2. Select the degree of fit.
    The second line in the drift preview shows the fitted polynomial.
    A degree of 0 will use the raw data.

3. Select normalisation method.
    The available options allow normalisation to wither the minimum or maximum
    signal in the drift area.


.. figure:: ../images/tutorial_drift_post.png
    :width: 400px
    :align: center

    The same image after drift compensation.
