#ifndef ___CTILE_PY_IF_H__
#define ___CTILE_PY_IF_H__
typedef struct {
    PyObject_HEAD
    CTile *c_tile;
    int type;
    PyObject* shape;
    PyObject* dtype;
} TileBase;
#endif
