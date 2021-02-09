Example: Importing File-Per-Line Data
=====================================

The `Import Wizard` allows users to provide specific options when importing data.
In this example demonstrates import of a modified TOFWERK instrument export with the file names in :ref:`table_tofwerkdata`.
The process will be similar for other programs that export lines as a directory of separate files.
The directory to be imported contains the files in :ref:`table_tofwerkdata`.
See also :ref:`Import Wizard`.


.. table:: Export filenames.
   :name: table_tofwerkdata
   :align: center

   +------------------------------------------------------------+
   | .../TOFWERK data/TW_Image3_25um2018.12.05-12h17m15s_AS.csv |
   +------------------------------------------------------------+
   | .../TOFWERK data/TW_Image3_25um2018.12.05-12h17m49s_AS.csv |
   +------------------------------------------------------------+
   | .../TOFWERK data/TW_Image3_25um2018.12.05-12h18m19s_AS.csv |
   +------------------------------------------------------------+
   | 474 more files...                                          |
   +------------------------------------------------------------+
   | .../TOFWERK data/TW_Image3_25um2018.12.05-16h23m13s_AS.csv |
   +------------------------------------------------------------+
   | .../TOFWERK data/dummy_file.csv                            |
   +------------------------------------------------------------+

.. table:: Export structure.
   :name: table_tofwerkstruct
   :align: center

   +---------------+-----------+-----------+-----+
   | t_elapsed_Buf | '[23Na]+' | '[24Mg]+' | ... |
   +===============+===========+===========+=====+
   | 0             | 1287.785  | 13.095236 | ... |
   +---------------+-----------+-----------+-----+
   | 0.10057113011 | 1664.5214 | 52.331398 | ... |
   +---------------+-----------+-----------+-----+
   | ...           | ...       | ...       | ... |
   +---------------+-----------+-----------+-----+
   | 25.9500810667 | 202654.8  | 1448.4713 | ... |
   +---------------+-----------+-----------+-----+
   | End of file.                                |
   +---------------+-----------+-----------+-----+

1. Select the `CSV Lines` format.
    This sets up the import wizard for CSV-as-lines import.

2. Select the path to the data.
    Either drag-and-drop or use the `Open Directiory` dialog to select the *TOFWERK data* directory.
    The number of `Matching files` is now 479, one more than the 478 lines.

2. Select import options.
    The file delimiter is ','.
    Looking at the :ref:`table_tofwerkstruct` we can see the is no header and one footer line,
    the `Footer Rows` should be set to 1.

3. Select the `File Regex`.
    The default regex ``.*\.csv`` will import all files ending with '.csv'.
    In our example there is a dummy file ``dummy_file.csv`` we do not want to import,
    we can set the regex to also look for 'TW_Image3' at the start of the filenames, ``TW_Image3.*\.csv``.
    The number of `Matching files` is now 478.

3. Select the `Sorting`.
    The exports in this example are marked with a timestamp,
    in most cases simple alphabetical sorting will work but we can also use the 'Timestamp option'.
    Set `Sorting` to timestamp and the sort key to ``%Y.%m.%d-%Hh%Mm%Ss``.

3. Select laser parameters and isotopes for import.
    Input the laser parameters.
    Non data names such as 't_elapsed_Buf' can be removed by pressing the `Edit Names` button.
