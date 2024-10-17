#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

uint8_t polygonf_contains_point(PyArrayObject *poly, double x, double y) {
  npy_intp n = PyArray_DIM(poly, 0);

  if (n < 3)
    return 0;

  double px, py, lx, ly;
  uint8_t flag, lflag;
  uint8_t inside = 0;

  lx = *(double *)PyArray_GETPTR2(poly, n - 1, 0);
  ly = *(double *)PyArray_GETPTR2(poly, n - 1, 1);
  lflag = ly >= y;

  for (npy_intp i = 0; i < n; ++i) {
    px = *(double *)PyArray_GETPTR2(poly, i, 0);
    py = *(double *)PyArray_GETPTR2(poly, i, 1);
    flag = py >= y;

    if (lflag != flag) {
      if (((py - y) * (lx - px) >= (px - x) * (ly - py)) == flag) {
        inside ^= 1;
      }
    }

    lx = px;
    ly = py;
    lflag = flag;
  }

  return inside;
}

static PyObject *polyext_polygonf_contains_points(PyObject *self,
                                                  PyObject *args) {
  PyObject *in[2];
  PyArrayObject *poly, *points;

  if (!PyArg_ParseTuple(args, "OO", &in[0], &in[1]))
    return NULL;

  poly =
      (PyArrayObject *)PyArray_FROM_OTF(in[0], NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (poly == NULL)
    return NULL;
  points =
      (PyArrayObject *)PyArray_FROM_OTF(in[1], NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (points == NULL) {
    Py_XDECREF(in[0]);
    return NULL;
  }

  PyArrayObject *res;
  npy_intp dims[] = {PyArray_DIM(points, 0)};
  res = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT8);

  npy_intp n = PyArray_DIM(points, 0);

  for (npy_intp i = 0; i < n; ++i) {
    uint8_t *out = (uint8_t *)PyArray_GETPTR1(res, i);
    *out =
        polygonf_contains_point(poly, *(double *)PyArray_GETPTR2(points, i, 0),
                                *(double *)PyArray_GETPTR2(points, i, 1));
  }

  Py_DECREF(in[0]);
  Py_DECREF(in[1]);

  return (PyObject *)res;
}

static PyMethodDef polyext_methods[] = {
    {"polygonf_contains_points", polyext_polygonf_contains_points, METH_VARARGS,
     "Check if multiple points are within a float type polygon."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef polyextmodule = {
    PyModuleDef_HEAD_INIT, "polyext_module", "Polygon extension modulde.", -1,
    polyext_methods};

PyMODINIT_FUNC PyInit_polyext(void) {
  PyObject *m;
  m = PyModule_Create(&polyextmodule);
  import_array();
  if (PyErr_Occurred())
    return NULL;
  return m;
}
