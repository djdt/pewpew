def formatIsotope(isotope: str, fstring: str = "{mass}{element}") -> str:
    element = isotope.strip("0123456789")
    mass = isotope.lower().strip("abcdefghijklmnopqrstuvwxyz")
    if element.isalpha() and mass.isdecimal():
        return fstring.format(element=element.capitalize(), mass=mass)
    else:
        return isotope
