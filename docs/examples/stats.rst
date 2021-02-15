Example: Mean of a region
=========================

|pewpew| implements two different tools for manual region selection,
the `Rectangle Selector` and `Lasso Selector`.
These tools function similarly to selection tools in other programs,
with regions selected by clicking and dragging on the image.

Holding **Shift** will **add** to the currently selected region while holding **Ctrl** will **subtract** from it.

.. figure:: ../images/tutorial_selection_lasso.png
    :width: 400px
    :align: center

    A series of regions selected using the `Lasso Selector` tool.
    The colour of selection depends on the users system colours.

1. Using the selection tools, select the desired region.
    See :ref:`Basic Usage` for details on the selection tools.

2. Open the `Statistics Dialog` in selection.
    Right clicking the selection will open its context menu.

3. Export the statistics using `Copy to Clipboard`.
    Statistics, including the means, can now be pasted into a spreadsheet program.
