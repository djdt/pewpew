Region Selection
================

Most elemental images have small regions of interest contained within the bulk of the image.
The selection tools in |pewpew| allow extraction of information from these regions.
The :ref:`Statistics` and :ref:`Colocalisation` dialogs both limit their calculations to selected regions and
a new image can be formed from the selection using **Edit -> Crop to Selection**.

Manual Selection
----------------

.. figure:: ../images/tutorial_selection_lasso.png
    :width: 200px
    :align: center

    A series of regions selected using the `Lasso Selector` tool.
    The colour of selection depends on the users system colours.

|pewpew| implements two different tools for manual region selection,
the `Rectangle Selector` and `Lasso Selector`.
These tools function similarly to selection tools in other programs,
with regions selected by clicking and dragging on the image.

Holding **shift** will **add** to the currently selected region while holding **control**
will **subtract** from it.


Selection Dialog
----------------

.. figure:: ../images/tutorial_selection_otsu.png
    :width: 200px
    :align: center

    Extraction of tissue from background using the `Selection Dialog` and Otsu's method.

The `Selection Dialog` selects a region based on a given threshold value or method.
The `Method` and `Comparison` combo-boxes selects the method for creating the thresholding value and
to compare the data to it.
As an example, if the `Method` produces a value of 1 and the `Comparison` is '>' then all values greater
than 1 will be selected.

Checking `Limit selection to current selection` will select the intersection (:math:`A \cap B`)
with the previously selected area.

Checking `Limit thresholding to selected value` will pass only currently selected values
to the thresholding method.
