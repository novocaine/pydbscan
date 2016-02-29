#include <Python.h>
#include "dbscan_nonsparse.h"
#include "dbscan_sparse.h"
#include <memory>
#include <numpy/arrayobject.h>

typedef struct {
    PyObject_HEAD
    bool is_double;
    // Use a union with a flag to be able to store either float or double 
    // dbscanner in the python object.
    //
    // could make this a unique_ptr, but it just complicates things given this
    // is a C API and we're obliged to manually delete it anyway.
    union {
        libdbscan::dbscan<float>* dbscanner_float;
        libdbscan::dbscan<double>* dbscanner_double;
    } dbscanner;
    PyObject* array;
} PyDbscan;

static void 
dbscan_dealloc(PyDbscan* self) {
    if (self->is_double) {
        delete self->dbscanner.dbscanner_double;
    } else {
        delete self->dbscanner.dbscanner_float;
    }
    Py_XDECREF(self->array);
}

void* get_c_corpus(PyObject* corpus, npy_intp* rows, npy_intp* cols, 
        int* type_num) {
    if (PyArray_Check(corpus)) {
        int nd = PyArray_NDIM(corpus);
        if (nd != 2) {
            PyErr_SetString(PyExc_TypeError, "Must be 2D array");
            return NULL;
        }

        PyArrayObject* array = ((PyArrayObject*)corpus);
        if (array->descr->type_num != PyArray_FLOAT && 
                array->descr->type_num != PyArray_DOUBLE) 
        {
            PyErr_SetString(PyExc_TypeError, 
                    "must be PyArray_FLOAT or PyArray_DOUBLE");
            return NULL;
        }

        *rows = array->dimensions[0];
        *cols = array->dimensions[1];
        *type_num = array->descr->type_num;
        return array->data;
    } else {
        PyErr_SetString(PyExc_TypeError, "corpus should be numpy array");
        return NULL;
    }
}

static PyObject*
dbscan_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyDbscan* self;
    self = (PyDbscan*)type->tp_alloc(type, 0);
    if (!self) {
        return NULL;
    }

    self->dbscanner.dbscanner_float = NULL;
    return (PyObject*)self;
}

static int 
dbscan_init(PyDbscan* self, PyObject* args, PyObject* kwds) {
    PyObject* corpus;
    const char* type = nullptr;
    const char* distance_metric = nullptr;
    if (!PyArg_ParseTuple(args, "O|ss", &corpus, &type, &distance_metric)) {
        PyErr_SetString(PyExc_TypeError, "couldn't parse args");
        return -1;
    }
    
    npy_intp rows, cols;
    int type_num;
    void* c_corpus = get_c_corpus(corpus, &rows, &cols, &type_num);
    if (!c_corpus) {
        return -1;
    }
    self->is_double = type_num == PyArray_DOUBLE;

    // incref the array because we're going to muck about with its data
    self->array = corpus;
    Py_XINCREF(self->array);

    // TODO - combinations are proliferating pretty badly here, maybe
    // make some sort of factory available in the lib available like
    // what we have in the CLI
    if (!type || !strcmp(type, "nonsparse")) {
        if (distance_metric && strcmp(distance_metric, "euclidean")) {
            Py_XDECREF(self->array);
            PyErr_SetString(PyExc_NotImplementedError,
                "only euclidean distance is supported for non-sparse arrays");
            return -1;
        }
        if (type_num == PyArray_FLOAT) {
            self->dbscanner.dbscanner_float = new libdbscan::dbscan_nonsparse<float>(
                    static_cast<float*>(c_corpus), rows, cols);
        } else {
            self->dbscanner.dbscanner_double = new libdbscan::dbscan_nonsparse<double>(
                    static_cast<double*>(c_corpus), rows, cols);
        }
    } else {
        if (!distance_metric || !strcmp(distance_metric, "euclidean")) {
            if (type_num == PyArray_FLOAT) {
                self->dbscanner.dbscanner_float = new libdbscan::dbscan_sparse<float>(
                        static_cast<float*>(c_corpus), rows, cols);
            } else {
                self->dbscanner.dbscanner_double = new libdbscan::dbscan_sparse<double>(
                        static_cast<double*>(c_corpus), rows, cols);
            }
        } else if (!strcmp(distance_metric, "cosine")) {
            if (type_num == PyArray_FLOAT) {
                self->dbscanner.dbscanner_float = new libdbscan::dbscan_sparse<float, 
                    libdbscan::cosine_similarity_metric<float> >(
                        static_cast<float*>(c_corpus), rows, cols);
            } else {
                self->dbscanner.dbscanner_double = new libdbscan::dbscan_sparse<double, 
                    libdbscan::cosine_similarity_metric<double> >(
                        static_cast<double*>(c_corpus), rows, cols);
            }
        } else {
            Py_XDECREF(self->array);
            PyErr_SetString(PyExc_NotImplementedError, "unknown distance metric");
            return -1;
        }
    }

    return 0;
}

static PyObject*
PyDbscan_run(PyDbscan* self, PyObject* args, PyObject* kwds) 
{
    float eps; int min_pts;
    if (!PyArg_ParseTuple(args, "fi", &eps, &min_pts)) {
        return NULL;
    }

    // TODO: this could be done with a numpy array and it would 
    // probably be a lot faster, but most of the time is spent
    // in the algo itself
    libdbscan::index_t num_rows;

    std::vector<libdbscan::index_t> results;
    std::vector<libdbscan::index_t> noise;
    try {
        if (self->is_double) {
            self->dbscanner.dbscanner_double->run(eps, min_pts, results, noise);
            num_rows = self->dbscanner.dbscanner_double->get_num_rows();
        } else {
            self->dbscanner.dbscanner_float->run(eps, min_pts, results, noise);
            num_rows = self->dbscanner.dbscanner_float->get_num_rows();
        }
    } catch (std::bad_alloc&) {
        PyErr_SetString(PyExc_MemoryError, "Alloc failure in run()");
        return NULL;
    } catch (...) {
        // we don't explicitly throw any exceptions in run(), but lets catch
        // anyway incase stdlib throws
        PyErr_SetString(PyExc_RuntimeError, "unknown exception in run()");
        return NULL;
    }

    PyObject* list_result = PyList_New(num_rows);
    if (!list_result) {
        PyErr_SetString(PyExc_RuntimeError, "couldn't create result");
        return NULL;
    }

    for (size_t i=0; i < num_rows; i++) {
        PyObject* pylong = PyLong_FromLong(results[i]);
        if (!pylong) {
            Py_DECREF(list_result);
            return NULL;
        }
        PyList_SET_ITEM(list_result, i, pylong);
    }

    return list_result;
}

static PyMethodDef dbscan_methods[] = {
    {"run", (PyCFunction)PyDbscan_run, METH_VARARGS, 
     "run(eps, min_pts, type) where eps is a float, min_pts is an integer, and"
     " type is 'nonsparse' (the default) or 'sparse'"
    },   
    {NULL}  /* Sentinel */
};

static PyTypeObject dbscanType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "dbscan.dbscan",           /*tp_name*/
    sizeof(PyDbscan),          /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)dbscan_dealloc,/*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "dbscan objects",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    dbscan_methods,             /* tp_methods */
    0,
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)dbscan_init,      /* tp_init */
    0,                         /* tp_alloc */
    dbscan_new,                 /* tp_new */
};


PyMODINIT_FUNC
initdbscan(void)
{
    if (PyType_Ready(&dbscanType) < 0) {
        return;
    }

    PyObject* module = Py_InitModule3("dbscan", dbscan_methods, 
            "DBSCAN algorithm");

    Py_INCREF(&dbscanType);
    PyModule_AddObject(module, "dbscan", (PyObject*)&dbscanType);

    import_array();
}
