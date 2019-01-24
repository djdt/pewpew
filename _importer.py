import inspect
import os.path
import pkgutil
import sys
from importlib import import_module


def import_submodule_class(package, type):
    for (_, name, _) in pkgutil.walk_packages([os.path.dirname(__file__)]):

        imported_module = import_module('.' + name, package=__name__)

        for i in dir(imported_module):
            attribute = getattr(imported_module, i)

            if inspect.isclass(attribute) and issubclass(attribute, type):
                setattr(sys.modules[__name__], i, attribute)
