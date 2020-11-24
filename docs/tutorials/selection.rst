Region Selection
================

Most elemental images have small regions of interest contained within the bulk of the image.
The selection tools in |pewpew| allow extraction of information from these regions.

Manual Selection
================

|pewpew| implements two different tools for manual region selection,
the `Rectangle Selector` and `Lasso Selector`.
These tools function similarly to selection tools in other programs,
with regions selected by clicking and dragging on the image.
Holding **shift** will **add** to the currently selected region while holding **control**
will **subtract** from it.


Selection Dialog
================

The `Selection Dialog` selects a region based on a given threshold value or method.
The `Method` combo-box selects the threshold method,
if `Manual` then the number in `Value` is used.

Checking `Limit selection to current selection` will select the intersection (:math:`A \cup B`)
with the previously selected area.

Checking `Limit thresholding to selected value` will pass only currently selected values
to the thresholding method.
