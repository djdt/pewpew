import matplotlib.transforms as mtransforms


def coords2index(im, x, y, inverted=False):
    """
    Converts data coordinates to index coordinates of the array.
    Parameters
    -----------
    im : An AxesImage instance
        The image artist to operation on
    x : number
        The x-coordinate in data coordinates.
    y : number
        The y-coordinate in data coordinates.
    inverted : bool, optional
        If True, convert index to data coordinates instead of
        data coordinates to index.
    Returns
    --------
    i, j : Index coordinates of the array associated with the image.
    """
    xmin, xmax, ymin, ymax = im.get_extent()
    if im.origin == "upper":
        ymin, ymax = ymax, ymin
    data_extent = mtransforms.Bbox([[ymin, xmin], [ymax, xmax]])
    array_extent = mtransforms.Bbox([[0, 0], im.get_array().shape[:2]])
    trans = mtransforms.BboxTransformFrom(data_extent) + mtransforms.BboxTransformTo(
        array_extent
    )

    if inverted:
        trans = trans.inverted()

    return trans.transform_point([y, x]).astype(int)


def coords2value(im, x, y, inverted=False):
    ix, iy = coords2index(im, x, y, inverted)
    return im.get_array()[ix, iy]
