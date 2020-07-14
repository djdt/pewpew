Misc
----
Standards result box should be in 2 rows.

Colocal is slow (coastes has many python loop).

Edit tool needs indicator for what is getting updated (single / multiple isotopes?)

Edit isotopes on right click.


Edit Tool
---------
_The transform tool is hard to use for trimming, draggable would help here._

Transform tool:
    * Draggable / sliding trim
    * Shifting data (mirror / edge / reflect) for isotope alignment


Draw Tool
---------
_Implement a tool for drawing_


~~Tool Rebuild~~
------------
_Currently the tool system opens a new view for the tool, this is really not required since access to the original laser and tool at once should not be needed. Instead, when the tool opens the laser view should close (move to tool) and then back to laser once tool is done._
Requires:
    * ~~Remove tool widget selection and button~~
    * ~~Add 'OK' button to return to widget.~~
    * ~~Implement the move, zoom and status bar updates in tool.~~
        * ~~These should be added to InteractiveCanvas (with move/scroll_button as None/0).~~
    * ~~Close tab button should revert to original image (as cancel does)~~

Info Dialog Rebuild
-------------------
_Dialogs for displaying information like the colocalisation and stats dialog are blocking. Either the stop blocking or are changed from dialogs to views. If non-blocking then they must update correctly and show which laser they are tied to, if views then open a new view and update on current widget change._

Non-blocking:
    _This is the easier version_
    * Modeless dialogs
    * Pass name to the dialog.
    * Update on selection changed

View:
    _This would be cool_
    * Open a new view / widget
    * Update on current widget change
        * How is tool selected / current widget stats handled?
    * Update on isotope changed
    * Update on selection changed

Import Wizard
-------------
_The current import dialogs should be consolidated into a single wizard, with more options. For example the agilent wizard page could look for and verify batchlogs and acqmethod then let you choose if you want to use them. All wiards could allow you to import specific isotopes / rename or reorder them._
