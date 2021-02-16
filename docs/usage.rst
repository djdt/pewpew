Basic Usage
===========
.. index:: Basic Usage

.. figure:: ./images/usage_guide.png
   :align: center

   An open image in |pewpew|. (1) Open image tabs, double click to edit laser name. (2) View controls. (3) Current image name, double click to rename. (4) Scalebar. (5) A selected region. (6) Image right click context menu. (7) Color scale, units on right if available. (8) Selectoin tools and widgets. (9) Current name selection. (10) Cursor unit, (μ = μm, i = pixel).


The first step in using |pewpew| is to load or import data.
For an example of using the import wizard see :ref:`Example: Importing file-per-line data` or read :ref:`Import Wizard`.
The image can be navigated by click and dragging with either the left or middle mouse button and using the scroll wheel.
The layout of images is able to be customised using the View controls (2).

.. figure:: ./images/usage_view_controls.jpg
   :align: center
   :width: 300 px
    
   View controls allow visualisation of multiple images at once.

Import and export controls are located in the **File** menu,
parameters and image transforms in the **Edit** menu and tools for calibration, processing and visualisation of data in the **Tools** menu bar.

The **View** menu contains options for customising the visual style of images.
Here you will find controls for the size and visibility of text, the colortable and image smoothing.
Color table ranges are individually editable for each open element via **View -> Colortable -> Set Range** or **Ctrl+R**.
A range of perceptually uniform (or near) colortables are available.

Selections
~~~~~~~~~~

|pewpew| implements two different tools for manual region selection,
the `Rectangle Selector` and `Lasso Selector`.
These tools function similarly to selection tools in other programs,
with regions selected by clicking and dragging on the image.

Holding **Shift** will **add** to the currently selected region while holding **Ctrl** will **subtract** from it.
