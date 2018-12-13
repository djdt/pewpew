def formatIsotopeTex(isotope):
    element = isotope.translate(None, "0123456789")
    mass = isotope.translate(
        None, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    return f"$^{{{mass}}}${element}"
