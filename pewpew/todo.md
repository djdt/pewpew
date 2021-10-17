Bugs
----

Multi
-----
_Current viewspaces are not utilised, remove viewspaces and use a single tabbed widget. The new widgets will allow multiple images per tab. Will need rewrite of View, Viewpsace, LaserView/Viewpsace and tying of images to LaserWidget_

Remove viewspace / views. One tabbed widget.

Allow multiple images to be open in one overlay view, see merge tool
Allow movement of images, see merge tool

A way to move each image to own tab
A way to combine image into current tab

Image alpha

Right-click context menus per graphics image.

Misc
----

Colocal is slow (coastes has many python loop).

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

