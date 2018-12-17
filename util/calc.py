import numpy as np

def weighted_polyfit(xs, ys, weights=None):
    coeffs, residuals = np.polyfit(xs, ys, 1, w=weights, full=True)
    p = np.poly1d(coeffs)
    yhat = p(xs)                         # or [p(z) for z in x]
    ybar = np.sum(ys)/len(ys)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((ys - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    return coeffs, ssreg / sstot, residuals


if __name__ == "__main__":
    a = np.array([3, -0.5, 2, 7])
    b = np.array([2.5, 0, 2, 8])
    w = np.array([1, 5, 1, 2])

    print('good', weighted_polyfit(a, a))
    print('bad', weighted_polyfit(a, b))
    print('bad weighted', weighted_polyfit(a, b, w))
