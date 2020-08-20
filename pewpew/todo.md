Bugs
----

Laser not set changed on edit tool

Misc
----

Overlay tool could use a colorbar scale.
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

