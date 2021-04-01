Import Wizard
=============
.. index:: Import Wizard

* **File -> Import -> Import Wizard**

.. table::  Drag-and-drop formats supported by |pewpew|.
    :align: center
    :name: import_formats

    +-------------+-------------+--------------+----------------+
    | Vendor      | Software    | Format       | Tested With    |
    +=============+=============+==============+================+
    | Agilent     | Mass Hunter | .b directory | 7500,7700,8900 |
    +-------------+-------------+--------------+----------------+
    | PerkinElmer |             | .xl files    |                |
    +-------------+-------------+--------------+----------------+
    | Thermo      | Qtegra      | .csv         | iCAP RQ        |
    +-------------+-------------+--------------+----------------+

|pewpew| supports drag-and-drop of specific formats (:ref:`import_formats`) and text-images with the extension '.csv' or '.txt'.
If drag-and-drop of a format is not supported or if special import requirements are needed then the import wizard should be used.
The `Import Wizard` allows users to provide specific options (:ref:`import_options`) when importing data.
The wizard consists of three pages:
format selection, path and options, and laser parameters.
For programs that export lines as a directory of separate files the '.csv' import option should be used.

.. _Kriss-Kross: https://doi.org/10.1021/acs.analchem.9b02380
.. _Regex: https://docs.python.org/3.3/howto/regex.html
.. _strftime: https://manpages.debian.org/buster/manpages-dev/strftime.3.en.html


.. table:: Format options available in the Import Wizards.
   :align: center
   :name: import_options

   +-------------+----------------------+-------------------------------------------------+
   | Format      | Option               | Description                                     |
   +=============+======================+=================================================+
   | Agilent     | Data File Collection | The method by which .d files are found.         |
   |             |                      |                                                 |
   |             |                      | If batch logs are malformed use Alphabetical.   |
   +             +----------------------+-------------------------------------------------+
   |             | Read names from Acq. | Read isotope names from the method.             |
   +-------------+----------------------+-------------------------------------------------+
   | CSV Lines   | Delimiter            | Delimter between values.                        |
   +             +----------------------+-------------------------------------------------+
   |             | Header / Footer Rows | Number of rows in file before / after the data. |
   +             +----------------------+-------------------------------------------------+
   |             | File Regex           | Regex_ used to find file names.                 |
   +             +----------------------+-------------------------------------------------+
   |             | Sorting              | Alphabetical sorts normally.                    |
   |             |                      |                                                 |
   |             |                      | Numerical sorts by numbers in the name only.    |
   |             |                      |                                                 |
   |             |                      | Timestamp using sortkey in strftime_ format.    |
   +-------------+----------------------+-------------------------------------------------+
   | Text Image  | Isotope Name         | Name of imported element.                       |
   +-------------+----------------------+-------------------------------------------------+
   | Thermo iCap | Export Format        | Data in rows or columns.                        |
   +             +----------------------+-------------------------------------------------+
   |             | Delimiter            |  Delimter between values.                       |
   +             +----------------------+-------------------------------------------------+
   |             | Decimal              | Character used as decimal place.                |
   +             +----------------------+-------------------------------------------------+
   |             | Use Analog           | Import analog data instead of counts.           |
   +-------------+----------------------+-------------------------------------------------+


Spot-wise Import Wizard
-----------------------
.. index:: Spot-wise Import Wizard

* **File -> Import -> Spotwise Import Wizard**

This wizard allows the import data collected in a spot-wise manner.
Imported data files are joined into a single continuous signal which is then used to find
and integrate peaks. Peaks can be detected using the algorithms in :ref:`peak_detection`.


.. table:: Peak detection algorithms and parameters.
   :align: center
   :name: peak_detection

   +----------------+-------------------+-----------------------------------------------------------------------------+
   | Algorithm      | Parameter         | Description                                                                 |
   +================+===================+=============================================================================+
   | Constant Value | Minimum value     | Continuous signals above this value are considered peaks.                   |
   +----------------+-------------------+-----------------------------------------------------------------------------+
   | CWT            | Min. / Max. width | CWT widths, should cover the expected peak width / 2.                       |
   +                +                   +                                                                             +
   |                | Width factor      | Peak width multiplier.                                                      |
   +                +                   +                                                                             +
   |                | Min. ridge SNR    | The minimum SNR for a CWT ridge to be considered a peak.                    |
   +                +                   +                                                                             +
   |                | Min. ridge length | The minimum continuous CWT ridge length.                                    |
   +----------------+-------------------+-----------------------------------------------------------------------------+
   | Moving window  | Window size       | Size of the rolling window.                                                 |
   +                +                   +                                                                             +
   |                | Window baseline   | Method for determining the signal baseline.                                 |
   +                +                   +                                                                             +
   |                | Window threshold  | Method for determining signal threshold.                                    |
   +                +                   +                                                                             +
   |                | Threshold         | Value used for thresholding.                                                |
   |                |                   |                                                                             |
   |                |                   | If method is 'Constant' the threshold is set to baseline + `Threshold`.     |
   |                |                   |                                                                             |
   |                |                   | If method is 'Std' the threshold is set to baseline + `Threshold` * stddev. |
   +----------------+-------------------+-----------------------------------------------------------------------------+


Kriss-Kross Import Wizard
-------------------------
.. index:: Kriss-Kross Import Wizard

* **File -> Import -> Kriss-Kross Import Wizard**

Import of Kriss-Kross_ collected Super-Resolution-Reconstruction images is performed
using the `Kriss-Kross Import Wizard`. This will guide users through import of the data
in a simliar manner to the :ref:`Import Wizard`.

.. seealso::
    :ref:`Example: Importing file-per-line data`.
     Example showing how to use the import wizard.
