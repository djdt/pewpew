import inspect
import os.path
import pkgutil
import sys
from PyQt5.QtWidgets import QDialog
from importlib import import_module

for (_, name, _) in pkgutil.walk_packages([os.path.dirname(__file__)]):

    imported_module = import_module('.' + name, package=__name__)

    for i in dir(imported_module):
        attribute = getattr(imported_module, i)

        if inspect.isclass(attribute) and issubclass(attribute, QDialog):
            setattr(sys.modules[__name__], i, attribute)
