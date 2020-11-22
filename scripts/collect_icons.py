#!/usr/bin/python
import os
import re

import argparse

from typing import Generator, List, Set, Tuple


def sorted_walk(root: str) -> Generator[Tuple[str, List[str], List[str]], None, None]:
    names = os.listdir(root)
    names.sort()

    dirs, files = [], []
    for name in names:
        if os.path.isdir(os.path.join(root, name)):
            dirs.append(name)
        else:
            files.append(name)

    yield root, dirs, files

    for d in dirs:
        path = os.path.join(root, d)
        if not os.path.islink(path):
            for x in sorted_walk(path):
                yield x


def collect_icons(path: str) -> Set[str]:
    regex_icon = "(?:fromTheme|qAction|qToolButton)\\(\\s*['\"]([a-z\\-]+)['\"]"
    icons = set()

    for root, _dirs, files in os.walk(path):
        for f in files:
            if not f.endswith(".py"):
                continue
            with open(os.path.join(root, f), "r") as fp:
                icons.update(re.findall(regex_icon, fp.read()))

    return icons


def write_qrc(qrc_path: str, icons_path: str, icons_root: str, icons: List[str]):
    with open(qrc_path, "w") as fp:
        fp.write('<!DOCTYPE RCC><RCC version="1.0">\n')

        for root, _, files in sorted_walk(icons_path):
            root = os.path.abspath(root)
            files = [
                f
                for f in files
                if os.path.splitext(f)[0] in icons or f == "index.theme"
            ]
            if len(files) == 0:
                continue
            fp.write(
                f'<qresource prefix="{os.path.sep + os.path.relpath(root, icons_root)}">\n'
            )
            for f in files:
                fp.write(f'<file alias="{f}">{os.path.join(root, f)}</file>\n')
            fp.write("</qresource>\n")

        fp.write("</RCC>")


script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
icons = list(collect_icons(script_path))

write_qrc("icons.qrc", "/usr/share/icons/breath2/", "/usr/share", icons)
