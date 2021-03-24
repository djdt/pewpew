Bugs
----

Misc
----

> Overlay tool could use a colorbar scale.
Cant make a 3 color color scale.

Colocal is slow (coastes has many python loop).

Edit tool needs indicator for what is getting updated (single / multiple isotopes?)

Edit Tool
---------


Draw Tool
---------
_Implement a tool for drawing_

Image Export Tool
-----------------
_tool for composition of images_


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

