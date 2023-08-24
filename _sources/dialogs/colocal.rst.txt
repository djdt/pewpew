Colocalisation Dialog
=====================
.. index:: Colocalisation Dialog

* **Image Context Menu -> Colocalisation**
* **Selection Context Menu -> Colocalisation**

Colcalisation can be used to quantify the spatial relationship between two elements.
The left side of the dialogs show the calculated colocalisation coefficients.
Both the 'Li ICQ' and 'Pearson R' coefficients are calculated over the entire image while 'Manders' are calculated using the Costes_ method.
The probability (ρ) of the Pearson R can be calculated by clicking the button next to it.
This is calculated by comparing the R value to those of random shuffles of the input.

The `Colocalisation Dialog` can also be opened from the context menu of a selected region.
This will limit the inputs to that region.

.. figure:: ../images/tutorial_colocal_plot.png
    :name: colocal_dialog
    :align: center

    The Colocalisation dialog.


.. _Costes: https://doi.org/10.1529/biophysj.103.038422
