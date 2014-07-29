#include <Python.h>
#include <structmember.h>
#include "cextent.h"
#include <iostream>


#define RETURN_IF_NULL(val) \
do {\
    if (val == NULL) { \
        return NULL; \
    } \
} while (0) 

#define RETURN_CHECK(val) \
do {\
    if (val == NULL) { \
        return NULL; \
    } else { \
        return val; \
    } \
} while (0) 


typedef struct {
    PyObject_HEAD
    unsigned ndim;
    unsigned long long size;
    CExtent *c_ex;
} TileExtent;

static bool 
is_integer(PyObject *o) {
    if (PyLong_Check(o) || PyInt_Check(o)) {
        return true;
    }
    return false;
}

/* Can't be defined here because it need &TileExtentType. */
static PyObject* _TileExtent_create_helper(CExtent *c_ex, bool return_none_if_null);

static long long 
get_longlong(PyObject *o) {
    if (PyNumber_Check(o)) {
        PyObject *_long;
        long long ret;

        _long = PyNumber_Long(o);
        ret = PyLong_AsLongLong(_long);
        Py_DECREF(_long);
        return ret;
    } else {
        assert(0);
    }
}

static PyObject*
_TileExtent_gettuple(TileExtent *self, unsigned long long *array, unsigned ndim)
{
    PyObject *tuple = PyTuple_New(ndim);
    RETURN_IF_NULL(tuple);

    for (int i = 0; i < ndim; i++) {
        PyTuple_SetItem(tuple, i, PyLong_FromLongLong(array[i]));   
    }

    return tuple;
}

static int
_TileExtent_settuple(TileExtent *self, 
                     PyObject *tuple, 
                     unsigned long long *dest, 
                     bool init)
{
    int i;
    PyObject *o;

    if (!PyTuple_Check(tuple)) {
        PyErr_SetString(PyExc_TypeError, "value sould be a tuple");
        return -1;
    }

    for (i = 0; i < self->ndim; i++) {
        o = PyTuple_GET_ITEM(tuple, i);
        dest[i] = get_longlong(o);
    }

    if (init) {
        self->c_ex->init_info();
        self->size = self->c_ex->size;
    }

    return 0;
}

static PyObject*
TileExtent_getul(TileExtent *self, void *closure)
{
    return _TileExtent_gettuple(self, self->c_ex->ul, self->c_ex->ndim);
}

static int
TileExtent_setul(TileExtent *self, PyObject *value, void *closure)
{
    return _TileExtent_settuple(self, value, self->c_ex->ul, true);
}

static PyObject*
TileExtent_getlr(TileExtent *self, void *closure)
{
    return _TileExtent_gettuple(self, self->c_ex->lr, self->c_ex->ndim);
}

static int
TileExtent_setlr(TileExtent *self, PyObject *value, void *closure)
{
    return _TileExtent_settuple(self, value, self->c_ex->lr, true);
}

static PyObject*
TileExtent_getshape(TileExtent *self, void *closure)
{
    return _TileExtent_gettuple(self, self->c_ex->shape, self->c_ex->ndim);
}

static int
TileExtent_setshape(TileExtent *self, PyObject *value, void *closure)
{
    return -1;
}

static PyObject*
TileExtent_getarray_shape(TileExtent *self, void *closure)
{
    if (self->c_ex->has_array_shape) {
        return _TileExtent_gettuple(self, self->c_ex->array_shape, self->c_ex->ndim);
    } else {
        Py_RETURN_NONE;
    }
}

static int
TileExtent_setarray_shape(TileExtent *self, PyObject *value, void *closure)
{
    return _TileExtent_settuple(self, value, self->c_ex->array_shape, false);
}

static void
TileExtent_dealloc(PyObject *o)
{
    TileExtent *self = (TileExtent*) o; 
    delete self->c_ex;
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
TileExtent_richcompare(PyObject* o, PyObject *o_other, int op)
{
    TileExtent *self = (TileExtent*) o;
    TileExtent *other = (TileExtent*) o_other;
    int ul_result, lr_result;
   
    ul_result = lr_result = 0;
    if (o_other == Py_None) {
        ul_result = lr_result = -1;
    } else {
        for (int i = 0; i < self->c_ex->ndim; i++) {
            if (self->c_ex->ul[i] > other->c_ex->ul[i])
                ul_result = 1;
            else if(self->c_ex->ul[i] < other->c_ex->ul[i])
                ul_result = -1;

            if (self->c_ex->lr[i] > other->c_ex->lr[i])
                lr_result = 1;
            else if(self->c_ex->lr[i] < other->c_ex->lr[i])
                lr_result = -1;
        }
    }
    switch (op) {
    case Py_GT:
        if (ul_result > 0)
            Py_RETURN_TRUE;
        Py_RETURN_FALSE;
    case Py_LT:
        if (ul_result < 0)
            Py_RETURN_TRUE;
        Py_RETURN_FALSE;
    case Py_EQ:
        if (ul_result == 0 and lr_result == 0)
            Py_RETURN_TRUE;
        Py_RETURN_FALSE;
    case Py_NE:
        if (ul_result != 0 or lr_result != 0)
            Py_RETURN_TRUE;
        Py_RETURN_FALSE;
    }
}

static PyObject *
TileExtent_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    TileExtent *self;

    self = (TileExtent*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->ndim = 0;
        self->size = 0;
    }

    return (PyObject *) self;
}

static PyObject*
TileExtent_reduce(PyObject* o)
{
    TileExtent *self = (TileExtent*) o;
    PyObject *mod, *result, *tuple, *obj, *ul, *lr, *array_shape;

    //std::cout << "__reduce__" << std::endl;
    result = PyTuple_New(2);
    tuple = PyTuple_New(3);
    mod = PyImport_ImportModule("spartan.array.extent");
    ul = TileExtent_getul(self, NULL);
    lr = TileExtent_getlr(self, NULL);
    array_shape = TileExtent_getarray_shape(self, NULL);

    if (result == NULL or tuple == NULL or mod == NULL or 
        ul == NULL or lr == NULL or array_shape == NULL) {
        if (result != NULL) Py_DECREF(result);
        if (tuple != NULL) Py_DECREF(tuple);
        if (mod != NULL) Py_DECREF(mod);
        if (ul != NULL) Py_DECREF(ul);
        if (lr != NULL) Py_DECREF(lr);
        if (array_shape != NULL) Py_DECREF(array_shape);
        return NULL;
    } 

    obj = PyObject_GetAttrString(mod, "create");
    Py_DECREF(mod);
    PyTuple_SetItem(tuple, 0, ul);
    PyTuple_SetItem(tuple, 1, lr);
    PyTuple_SetItem(tuple, 2, array_shape);
    PyTuple_SetItem(result, 0,  obj);
    PyTuple_SetItem(result, 1,  tuple);

    return result;
}

static PyObject *
TileExtent_to_slice(PyObject* o)
{
    TileExtent *self = (TileExtent*) o;
    PyObject *result;

    //std::cout << __func__ << std::endl;
    result = PyTuple_New(self->c_ex->ndim);
    RETURN_IF_NULL(result);

    for (int i = 0; i < self->c_ex->ndim; i++) {
        PyObject *slc;
        slc = PySlice_New(PyLong_FromLongLong(self->c_ex->ul[i]),
                          PyLong_FromLongLong(self->c_ex->lr[i]),
                          NULL);
        RETURN_IF_NULL(slc);
        PyTuple_SetItem(result, i, slc);
    }

    return result;
}

static PyObject *
TileExtent_ravelled_pos(PyObject* o)
{
    TileExtent *self = (TileExtent*) o;
    unsigned long long rpos;
    PyObject *result;

    //std::cout << __func__ << std::endl;
    rpos = ravelled_pos(self->c_ex->ul, self->c_ex->array_shape, self->c_ex->ndim);
    result = PyLong_FromLongLong(rpos);
    RETURN_CHECK(result);
}

static PyObject *
TileExtent_to_global(PyObject *o, PyObject *args)
{
    TileExtent *self = (TileExtent*) o;
    PyObject *idx, *result;

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "O", &idx))
        return NULL;
    
    result = PyLong_FromLongLong(self->c_ex->to_global(get_longlong(idx)));
    RETURN_CHECK(result);
}

static long
TileExtent_hash(PyObject *o)
{
    TileExtent *self = (TileExtent*) o; 
    long ret = self->c_ex->to_global(0);
    //std::cout << __func__ << " " << ret << " " << self->c_ex->ul[0] << " " << self->c_ex->ul[1] << " " << self << std::endl;
    return ret;
}

static PyObject*
TileExtent_add_dim(PyObject* o)
{
    TileExtent *self = (TileExtent*) o;

    //std::cout << __func__ << std::endl;
    return _TileExtent_create_helper(self->c_ex->add_dim(), false);
}

static PyObject *
TileExtent_clone(PyObject* o)
{
    TileExtent *self = (TileExtent*) o;

    //std::cout << __func__ << std::endl;
    return _TileExtent_create_helper(self->c_ex->clone(), false);
}

static PyMemberDef TileExtent_members[] = {
    {(char*)"ndim", T_INT, offsetof(TileExtent, ndim), 0, (char*)"number of dimenesions"},
    {(char*)"size", T_LONGLONG, offsetof(TileExtent, size), 0, (char*)"size"},
    {NULL} /* Sentinel */
};

static PyGetSetDef TileExtent_getseters[] = {
    {(char*)"ul", (getter)TileExtent_getul, (setter)TileExtent_setul, (char*)"ul", NULL},
    {(char*)"lr", (getter)TileExtent_getlr, (setter)TileExtent_setlr, (char*)"lr", NULL},
    {(char*)"shape", (getter)TileExtent_getshape, 
     (setter)TileExtent_setshape, (char*) "shape", NULL},
    {(char*)"array_shape", (getter)TileExtent_getarray_shape, 
     (setter)TileExtent_setarray_shape, (char*)"array_shape", NULL},
    {NULL} /* Sentinel */
};

static PyMethodDef TileExtent_methods[] = {
    {"__reduce__", (PyCFunction)TileExtent_reduce, METH_NOARGS,
     "__reduce__"},
    {"to_slice", (PyCFunction)TileExtent_to_slice, METH_NOARGS,
     "Respresent the extent by slices."},
    {"ravelled_pos", (PyCFunction)TileExtent_ravelled_pos, METH_NOARGS,
     "Ravel the extent."},
    {"to_global", (PyCFunction)TileExtent_to_global, METH_VARARGS,
     "Transform extent[idx] to the global index."},
    {"add_dim", (PyCFunction)TileExtent_add_dim, METH_NOARGS,
     "Add a dimension to this extent."},
    {(char*)"clone", (PyCFunction)TileExtent_clone, METH_NOARGS,
     (char*)"Clone the extent."},
    {NULL} /* Sentinel */
};

static PyTypeObject TileExtentType = {
    PyObject_HEAD_INIT(NULL)
    0,                           /* ob_size */
    "extent.TileExtent",         /* tp_name */
    sizeof(TileExtent),          /* tp_basicsize */
    0,                           /* tp_itemsize */
    TileExtent_dealloc,          /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    0,                           /* tp_compare */
    0,                           /* tp_repr */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    TileExtent_hash,             /* tp_hash */
    0,                           /* tp_call */
    0,                           /* tp_str */
    0,                           /* tp_getattro */
    0,                           /* tp_setattro */
    0,                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,          /* tp_flags */
    "extentobjects",             /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    TileExtent_richcompare,      /* tp_richcompare */
    0,                           /* tp_weaklistoffset */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    TileExtent_methods,          /* tp_methods */
    TileExtent_members,          /* tp_members */
    TileExtent_getseters,        /* tp_getset */
    0,                           /* tp_base */
    0,                           /* tp_dict */
    0,                           /* tp_descr_get */
    0,                           /* tp_descr_set */
    0,                           /* tp_dictoffset */
    0,                           /* tp_init */
    0,                           /* tp_alloc */
    TileExtent_new,              /* tp_new */
};

static PyObject*
_TileExtent_create_helper(CExtent *c_ex, bool return_none_if_null)
{
    TileExtent *ex;

    if (c_ex == NULL) {
        if (return_none_if_null) {
            Py_INCREF(Py_None);
            return Py_None;
        } else {
            return NULL;
        }
    } else {
        ex = PyObject_New(TileExtent, &TileExtentType);
        if (ex == NULL) {
            delete c_ex;
            return NULL;
        }

        ex->size = c_ex->size;
        ex->ndim = c_ex->ndim;
        ex-> c_ex = c_ex;
        return (PyObject*)ex;
    }
}



static PyObject*
create(PyObject *self, PyObject *args)
{
    unsigned long long c_ul[MAX_NDIM], c_lr[MAX_NDIM], c_array_shape[MAX_NDIM];
    int i, ndim;
    PyObject *ul, *lr, *array_shape;
    CExtent *c_ex; 
    TileExtent *ex;

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "OOO", &ul, &lr, &array_shape))
        return NULL;

    ndim = PySequence_Size(ul); 
    for (i = 0; i < ndim; i++) {
        c_ul[i] = get_longlong(PySequence_GetItem(ul, i));
        c_lr[i] = get_longlong(PySequence_GetItem(lr, i));
    }

    if (array_shape != Py_None) {
        for (i = 0; i < ndim; i++) {
            c_array_shape[i] = get_longlong(PySequence_GetItem(array_shape, i));
        }
        c_ex = extent_create(c_ul, c_lr, c_array_shape, ndim);
    } else {
        c_ex = extent_create(c_ul, c_lr, NULL, ndim);
    }
    return _TileExtent_create_helper(c_ex, true);
}

static PyObject*
from_shape(PyObject *self, PyObject *args)
{
    PyObject *list;

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "O", &list))
        return NULL;

    unsigned long long shape[MAX_NDIM];
    int ndim = PySequence_Size(list);
    for (int i = 0; i < ndim; i++) {
        shape[i] = get_longlong(PySequence_GetItem(list, i));
    }

    return _TileExtent_create_helper(extent_from_shape(shape, ndim), false);
}

static PyObject*
intersection(PyObject *self, PyObject *args)
{
    TileExtent *ex_a, *ex_b;
    CExtent *c_ex; 
    TileExtent *ex;

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "OO", (PyObject **)&ex_a, (PyObject **)&ex_b))
        return NULL;

    return _TileExtent_create_helper(intersection(ex_a->c_ex, ex_b->c_ex), true);
}

static PyObject*
compute_slice(PyObject *self, PyObject *args)
{
    PyObject *slice;
    TileExtent *ex, *ret_ex;
    CExtent *ret_c_ex;
    Slice c_slice[MAX_NDIM];
    int i;

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "OO", (PyObject **)&ex, &slice))
        return NULL;

    for (i = 0; i < ex->ndim; i++) {
        c_slice[i].start = 0;
        c_slice[i].stop = ex->c_ex->shape[i];
    }

    if (!PyTuple_Check(slice)) {
        if (PySlice_Check(slice)) {
            Py_ssize_t start, stop, step, slicelength;
            PySlice_GetIndicesEx((PySliceObject*)slice, ex->c_ex->shape[0], 
                                 &start, &stop, &step, &slicelength);
            c_slice[0].start = (long long) start;
            c_slice[0].stop = (long long) stop;
        } else if (slice == Py_None) {
            ;
        } else {
            c_slice[0].start = get_longlong(slice);
            c_slice[0].stop = c_slice[0].start + 1;
        }
    } else {
        for (i = 0; i < PyTuple_Size(slice); i++) {
            PyObject *slc = PyTuple_GET_ITEM(slice, i);
            if (PySlice_Check(slc)) {
                Py_ssize_t start, stop, step, slicelength;
                PySlice_GetIndicesEx((PySliceObject*)slc, ex->c_ex->shape[i], 
                                     &start, &stop, &step, &slicelength);
                c_slice[i].start = (long long) start;
                c_slice[i].stop = (long long) stop;
            } else if (slice == Py_None) {
                ;
            } else {
                c_slice[i].start = get_longlong(slc);
                c_slice[i].stop = c_slice[i].start + 1;
            }
        }
    }
    return _TileExtent_create_helper(compute_slice(ex->c_ex, c_slice, ex->ndim), true);
}

static PyObject*
offset_from(PyObject *self, PyObject *args)
{
    TileExtent *ex_a, *ex_b;

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "OO", (PyObject **)&ex_a, (PyObject **)&ex_b))
        return NULL;

    return _TileExtent_create_helper(offset_from(ex_a->c_ex, ex_b->c_ex), true);
}

static PyObject*
offset_slice(PyObject *self, PyObject *args)
{
    TileExtent *ex_a, *ex_b;
    Slice slice[MAX_NDIM];

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "OO", (PyObject **)&ex_a, (PyObject **)&ex_b))
        return NULL;

    offset_slice(ex_a->c_ex, ex_b->c_ex, slice);
    PyObject *tuple = PyTuple_New(ex_a->ndim);
    RETURN_IF_NULL(tuple);
    for (int i = 0; i < ex_a->ndim; i++) {
        PyObject *slc;
        slc = PySlice_New(PyLong_FromLongLong(slice[i].start),
                          PyLong_FromLongLong(slice[i].stop),
                          NULL);
        RETURN_IF_NULL(slc);
        PyTuple_SetItem(tuple, i, slc);
    }
    return tuple;
}

static PyObject*
from_slice(PyObject *self, PyObject *args)
{
    PyObject *slice, *shape_obj;
    Slice c_slice[MAX_NDIM];
    unsigned long long shape[MAX_NDIM];
    int i, ndim;

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "OO", (PyObject **)&slice, &shape_obj))
        return NULL;

    ndim = PySequence_Size(shape_obj);
    for (i = 0; i < ndim; i++) {
        shape[i] = get_longlong(PySequence_GetItem(shape_obj, i));
        c_slice[i].start = 0;
        c_slice[i].stop = shape[i];
    }

    if (!PyTuple_Check(slice)) {
        if (PySlice_Check(slice)) {
            Py_ssize_t start, stop, step, slicelength;
            int ret = PySlice_GetIndicesEx((PySliceObject*)slice, shape[0], 
                                           &start, &stop, &step, &slicelength);
            c_slice[0].start = (long long) start;
            c_slice[0].stop = (long long) stop;
        } else if (slice == Py_None) {
            ;
        } else {
            c_slice[0].start = get_longlong(slice);
            c_slice[0].stop = c_slice[0].start + 1;
        }
    } else {
        for (i = 0; i < PyTuple_Size(slice); i++) {
            PyObject *slc = PyTuple_GET_ITEM(slice, i);
            if (PySlice_Check(slc)) {
                Py_ssize_t start, stop, step, slicelength;
                PySlice_GetIndicesEx((PySliceObject*)slc, shape[i], 
                                     &start, &stop, &step, &slicelength);
                c_slice[i].start = (long long) start;
                c_slice[i].stop = (long long) stop;
            } else if (slice == Py_None) {
                ;
            } else {
                c_slice[i].start = get_longlong(slc);
                c_slice[i].stop = c_slice[i].start + 1;
            }
        }
    }
    return _TileExtent_create_helper(from_slice(c_slice, shape, ndim), false);
}

static PyObject*
drop_axis(PyObject *self, PyObject *args)
{
    TileExtent *ex;
    PyObject *axis;

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "OO", (PyObject **)&ex, &axis))
        return NULL;

    if (axis == Py_None) {
       return _TileExtent_create_helper(extent_create(NULL, NULL, NULL, 0), false);
    } else {
       return _TileExtent_create_helper(drop_axis(ex->c_ex, get_longlong(axis)), false);
    }
}

static PyObject*
index_for_reduction(PyObject *self, PyObject *args)
{
    //std::cout << __func__ << std::endl;
    return drop_axis(self, args);
}

/*
static PyObject*
shape_for_reduction(PyObject *self, PyObject *args)
{
    PyObject *shape, *axis;
    PyObject *list;

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "OO", &shape, &axis))
        return NULL;

    if (axis == Py_None) {
        return PyTuple_New(0);
    } else {
        list = PySequence_List(shape);
        RETURN_IF_NULL(list);
        if (PySequence_DelItem(list, get_longlong(axis))) {
            return NULL;
        }
        return list;
    }
}
*/

static PyObject*
find_shape(PyObject *self, PyObject *args)
{
    PyObject *list;
    TileExtent *ex;

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "O", &list))
        return NULL;

    int i, num_of_ex = PySequence_Size(list);
    CExtent **extents = new CExtent*[num_of_ex];
    unsigned long long shape[MAX_NDIM];
    for (i = 0; i < num_of_ex; i++) {
        ex = (TileExtent*) PySequence_GetItem(list, i);
        extents[i] = ex->c_ex;
    }
    find_shape(extents, num_of_ex, shape);
    list = PyTuple_New(ex->ndim);
    RETURN_IF_NULL(list);
    for (i = 0; i < ex->ndim; i++) {
        PyTuple_SetItem(list, i, PyLong_FromLongLong(shape[i]));   
    }
    return list;
}

/*
static PyObject*
shapes_match(PyObject *self, PyObject *args)
{
    TileExtent *ex_a, *ex_b;

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "OO", (PyObject **)&ex_a, (PyObject **)&ex_b))
        return NULL;

    if (shapes_match(ex_a->c_ex, ex_b->c_ex)) {
        std::cout << "TRUE" << std::endl;
        Py_RETURN_TRUE;
    } else {
        std::cout << "FALSE" << std::endl;
        Py_RETURN_FALSE;
    }
}
*/

static PyObject*
unravelled_pos(PyObject *self, PyObject *args)
{
    PyObject *idx_obj, *array_shape, *result;

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "OO", &idx_obj, &array_shape))
        return NULL;

    unsigned long long idx = get_longlong(idx_obj);
    unsigned ndim = PyTuple_Size(array_shape);

    result = PyTuple_New(ndim);
    RETURN_IF_NULL(result);
    for (int i = ndim - 1; i >= 0; i--) {
        unsigned long long item = get_longlong(PySequence_GetItem(array_shape, i));
        PyTuple_SetItem(result, i, PyLong_FromLongLong(idx % item));
        idx /= item;
    }

    return result;
}

static PyObject*
ravelled_pos(PyObject *self, PyObject *args)
{
    PyObject *idx, *array_shape, *result;

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "OO", &idx, &array_shape))
        return NULL;

    unsigned long long rpos = 0, mul = 1;
    unsigned ndim = PySequence_Size(array_shape);

    for (int i = ndim - 1; i >= 0; i--) {
        rpos += mul * get_longlong(PySequence_GetItem(idx, i));
        mul *= get_longlong(PySequence_GetItem(array_shape, i));
    }

    result = PyLong_FromLongLong(rpos);
    RETURN_CHECK(result);
}

static PyMethodDef extent_methods[] = {
    {"create", create, METH_VARARGS, ""},
    {"from_shape", from_shape, METH_VARARGS, ""},
    {"intersection", intersection, METH_VARARGS, ""},
    {"compute_slice", compute_slice, METH_VARARGS, ""},
    {"offset_from", offset_from, METH_VARARGS, ""},
    {"offset_slice", offset_slice, METH_VARARGS, ""},
    {"from_slice", from_slice, METH_VARARGS, ""},
    {"drop_axis", drop_axis, METH_VARARGS, ""},
    {"index_for_reduction", index_for_reduction, METH_VARARGS, ""},
    {"find_shape", find_shape, METH_VARARGS, ""},
    {"unravelled_pos", unravelled_pos, METH_VARARGS, ""},
    {"ravelled_pos", ravelled_pos, METH_VARARGS, ""},
   
    /**
     * These methods are noly used from Python and do not create 
     * any extent. Therefore, they don't have to be implemented
     * by the native Python API
     */

    //{"shapes_match", shapes_match, METH_VARARGS, ""},
    //{"shape_for_reduction", shape_for_reduction, METH_VARARGS, ""},
    //{"find_overlapping", find_overlapping, METH_VARARGS, ""},
    //{"all_nonzero_shape", all_nonzero_shape, METH_VARARGS, ""},
    //{"find_rect"}, find_rect, METH_VARARGS, ""},
    //{"is_complete", is_complete, METH_VARARGS, ""},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_cextent_py_if(void) 
{
    PyObject* m;

    TileExtentType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&TileExtentType) < 0)
        return;

    m = Py_InitModule3("_cextent_py_if", extent_methods,
                       "Python interface for cextent module");

    Py_INCREF(&TileExtentType);
    PyModule_AddObject(m, "TileExtent", (PyObject *)&TileExtentType);
}

