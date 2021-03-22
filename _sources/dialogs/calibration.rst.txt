Calibration Dialog
==================
.. index:: Calibration Dialog

* **Right Click -> Calibration**

The calibration dialog allows manual entry of the gradient, intercept and unit for each isotope.
Entry of specific concentrations, responses and weights is possible using the points table.
Changing points will automatically update the gradient, intercept and other coefficients.
To apply the current calibration to all other open images check the `Apply calibration to all images.` box and then click `Apply`.
The current (or all) calibrations can be copied to the clipboard using the copy button and pasted in open images using **Ctrl+V**.

Calibration Curve
-----------------
.. index:: Calibration Curve

Both the `Calibration Dialog` and :ref:`Standards Tool` have a button `Button Plot` that will open
a window with the current calibration plotted.

.. figure:: ../images/tutorial_calibration_plot.png
    :width: 400px
    :align: center

    Calibration curve, right clicking the plot allows copying the image to the clipboard.

.. seealso::
   :ref:`Standards Tool`
    Tool for the creation of calibrations from standards.
