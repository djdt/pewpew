Example: Thresholded division of elements
=========================================

Division of two elements is commonly used to normalise one element to another.
Without thresholding of low background values this can produce a rather unattractive image.
See also :ref:`Calculator`.

.. centered::
    |calc_img_bad| = |calc_img_z66| / |calc_img_p31|

In this example we use the calculator tool to simultaneously threshold and divide two elements.

.. centered::
    |calc_img_div| = |calc_img_z66| / |calc_img_p31|

.. |calc_img_z66| image:: ../images/tutorial_calc_zn66.png
    :width: 200px
.. |calc_img_p31| image:: ../images/tutorial_calc_p31.png
    :width: 200px
.. |calc_img_div| image:: ../images/tutorial_calc_div.png
    :width: 200px
.. |calc_img_bad| image:: ../images/tutorial_calc_baddiv.png
    :width: 200px


1. Determine the background value of the divisor image.
    In this example a value of 100 sufficed.

2. Using the calculator tool perform the operation.
    Enter ``if P31 > 100 then Zn66 / P31 else 0`` into the `Formula` box.

    The first part of the if/then/else masks the data so only values above the threshold are
    operated on. The second part performs the division while the last part sets unmasked data to 0.

3. Enter the new name, and click `Apply`.
    Choose a unique name to prevent overwriting data.

Example: Image Segmentation
===========================

Selection of a region of interest of structural feature can be performed using segmentation.
In this example we will use multi-level kmeans to select certain features from Fe56 of a
tissue-micro-array core.

.. centered::
   |calc_img_k_og|

1. Create the segmented image mask.
    Enter ``segment(Fe56, kmeans(Fe56, 3))`` into the `Formula` box.
    Here we are using 3 level kmeans clustering.

.. centered::
    |calc_img_k_mask|

2. Enter ``masked`` into the `Name` box, and click `Apply`.
    You can choose any name will work.

3. Select data from the original.
    Enter ``mask(Fe56, masked == 3)`` into the `Formula` box.
    The selected data is limited to regions where k==3.

.. centered::
    |calc_img_k_select|

3. Enter the new name, and click `Apply`.
    Choose a unique name to prevent overwriting data.


.. |calc_img_k_og| image:: ../images/tutorial_calc_kmean_fe.png
    :width: 300px
.. |calc_img_k_mask| image:: ../images/tutorial_calc_kmean_mask.png
    :width: 300px
.. |calc_img_k_select| image:: ../images/tutorial_calc_kmeans_selected.png
    :width: 300px
