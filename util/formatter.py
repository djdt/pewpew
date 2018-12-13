def formatIsotopeTex(isotope):
    element = isotope.strip("0123456789")
    mass = isotope.strip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return f"$^{{{mass}}}${element}"
