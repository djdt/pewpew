Importing Data
==============

.. table:: Vendor formats supported by |pewpew|.

    +-----------+-----------+-------------+--------------+
    |Vendor     |Software   |Format       |Tested With   |
    +===========+===========+=============+==============+
    |Agilent    |Mass Hunter|.b directory |7500,7700,8900|
    +-----------+-----------+-------------+--------------+
    |PerkinElmer|           |directory    |              |
    +-----------+-----------+-------------+--------------+
    |Thermo     |Qtegra     |.csv         |iCAP RQ       |
    +-----------+-----------+-------------+--------------+

For the majority of users importing data consists of dragging-and-dropping of files into |pewpew|.
An `Import Wizard` exists for cases of incorrectly formatted data or if you required finer control over an import.

Import Wizard
-------------

* **File -> Import -> Import Wizard**

Importing data using the `Import Wizard` begins by selecting the format to be used.
The path of the file or directory can be chosen on the next page using either the file dialog `Open File`
or by drag-and-drop of the file into the import wizard.

Once the file has been selected format options will be automatically filled with sensible defaults.
These can be changed if required, for example to extract the analog channel from a Thermo iCAP CSV.

The final page of the wizard allows the user to fill in any incorrect laser parameters and to rename or delete
imported isotopes. Clicking finish will import the data into |pewpew|.
