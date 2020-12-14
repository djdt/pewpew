Convolution
-----------

* **Tools -> Edit Tool -> Convolve / Deconvolve**

The `Convolve` and `Deconvolve` tools enable the creation of a kernel that can be
convolved with image data in one or two dimensions. For a full list of kernels see
:ref:`convolve_kernels`. The `Size` (in pixels) and `Scale` of each kernel is also
alterable.


Available Kernels
~~~~~~~~~~~~~~~~~

.. table:: Convolution kernels and parameters.
    :name: convolve_kernels
    :align: center

    +-------------------+-----------+-----------+-----------+
    | Kernel            | Parameters                        |
    +===================+===========+===========+===========+
    | Beta              | α [0, ∞)  | β [0, ∞)  |           |
    +-------------------+-----------+-----------+-----------+
    | Exponential       | λ [0, ∞)  |           |           |
    +-------------------+-----------+-----------+-----------+
    | Inverse-gamma     | α [0, ∞)  | β [0, ∞)  |           |
    +-------------------+-----------+-----------+-----------+
    | Laplace           | b [0, ∞)  | μ (-∞, ∞) |           |
    +-------------------+-----------+-----------+-----------+
    | Log-Laplace       | b [0, ∞)  | μ (-∞, ∞) |           |
    +-------------------+-----------+-----------+-----------+
    | Log-normal        | σ [0, ∞)  | μ (-∞, ∞) |           |
    +-------------------+-----------+-----------+-----------+
    | Gaussian (normal) | σ [0, ∞)  | μ (-∞, ∞) |           |
    +-------------------+-----------+-----------+-----------+
    | Super-Gaussian    | σ [0, ∞)  | μ (-∞, ∞) | P (-∞, ∞) |
    +-------------------+-----------+-----------+-----------+
    | Triangular        | a (-∞, 0] | b [0, ∞)  |           |
    +-------------------+-----------+-----------+-----------+


Example: Removing wash-out blur
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../images/tutorial_convolve_pre.png
    :width: 300px
    :align: center

    An image with wash-out blur, seen on the right hand side of the tissue.

1. Calculate the wash-out time in pixels.
    If the wash-out time is known then use wash-out (s) / acquisition time (s).

2. `Deconvolve` in one-dimension with a non-symmetrical point-spread-function.
    In this example `Log-Normal` (size=12; σ=1.50, μ=0.0) worked well.

3. Optional, to remove deconvolution artefacts take the absolute value of the image.
    Use the abs() function in the `Calculator`.


.. figure:: ../images/tutorial_convolve_post.png
    :width: 300px
    :align: center

    The same image post-deconvolution. Notice the lessen blur on the right hand side.
