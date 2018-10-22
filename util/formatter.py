import re

isore = re.compile(r"(\d*)([A-Z][a-z]*)(\d*)")


def isotopeFormat(isotope):

    m = isore.match(isotope)
    element = m.group(2)
    mass = m.group(1) if m.group(1) is not "" else m.group(3)

    return f"$^{{{mass}}}${element}"
