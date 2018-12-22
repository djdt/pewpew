def formatIsotope(isotope, fstring="{mass}{element}"):
    element = isotope.strip("0123456789")
    mass = isotope.lower().strip("abcdefghijklmnopqrstuvwxyz")
    if element.isalpha() and mass.isdecimal():
        return fstring.format(element.capitalize(), mass)
    else:
        return isotope
