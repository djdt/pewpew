ELEMENTS = [
    "Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au",
    "B", "Ba", "Be", "Bh", "Bi", "Bk", "Br",
    "C", "Ca", "Cd", "Ce", "Cf", "Cl", "Cm", "Co", "Cr", "Cs", "Cu",
    "Db", "Dy",
    "Er", "Es", "Eu",
    "F", "Fe", "Fm", "Fr",
    "Ga", "Gd", "Ge",
    "H", "He", "Hf", "Hg", "Ho", "Hs",
    "I", "In", "Ir",
    "K", "Kr", "La", "Li", "Lr", "Lu",
    "Md", "Mg", "Mn", "Mo", "Mt",
    "N", "Na", "Nb", "Nd", "Ne", "Ni", "No", "Np",
    "O", "Os",
    "P", "Pa", "Pb", "Pd", "Pm", "Po", "Pr", "Pt", "Pu",
    "Ra", "Rb", "Re", "Rf", "Rh", "Rn", "Ru",
    "S", "Sb", "Sc", "Se", "Sg", "Si", "Sm", "Sn", "Sr",
    "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl", "Tm",
    "U",
    "V",
    "W",
    "Xe",
    "Y", "Yb",
    "Zn", "Zr",
]


def formatIsotope(isotope: str, fstring: str = "{mass}{element}") -> str:
    element = isotope.strip("0123456789")
    mass = isotope.lower().strip("abcdefghijklmnopqrstuvwxyz")
    if element.isalpha() and mass.isdecimal():
        return fstring.format(element=element.capitalize(), mass=mass)
    else:
        return isotope
