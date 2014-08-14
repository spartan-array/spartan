#ifndef __CARRAY_H__
#define __CARRAY_H__
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include "../extent/cextent.h"

extern int npy_type_size[]; 
extern int npy_type_number[];
extern char npy_type_token[];

inline int
npy_type_token_to_size(char token)
{
    int i, type_size;
    for (i = 0, type_size = 0; npy_type_token[i] != -1; i++) {
        if (npy_type_token[i] == token) {
            type_size = npy_type_size[i];
            break;
        }         
    }
    assert(npy_type_token[i] != -1);
    return type_size;
}

inline int
npy_type_token_to_number(char token)
{
    int i, type_number;
    for (i = 0, type_number = 0; npy_type_token[i] != -1; i++) {
        if (npy_type_token[i] == token) {
            type_number = npy_type_number[i];
            break;
        }         
    }
    assert(npy_type_token[i] != -1);
    return type_number;
}

#include <map>
class NpyMemManager {
public:
    NpyMemManager(void) : ptr(NULL), own_by_npy(false) {};
    NpyMemManager(char* ptr, bool own_by_npy = false) {
        std::map<char*, int>::iterator it;

        this->ptr = ptr;
        this->own_by_npy = own_by_npy;
        if ((it = NpyMemManager::refcount.find(ptr)) == NpyMemManager::refcount.end()) {
            refcount[ptr] = 1; 
            if (own_by_npy) {
                Py_INCREF(ptr);
            }
        } else {
            it->second += 1;
        }
    };
    ~NpyMemManager(void) {
        int count = refcount[ptr]; 
        if (count == 1) {
            std::cout << "Now we are deleteing a data region " << own_by_npy << std::endl;
            refcount.erase(ptr);
            if (own_by_npy) {
                Py_DECREF(ptr);
            } else {
                delete ptr;
            }
            std::cout << "Now we are done" << std::endl;
        } else {
            refcount[ptr] -= 1;
        }
    };
    NpyMemManager operator=(const NpyMemManager& obj) {
        if (this != &obj) {
            this->ptr = obj.ptr;
            this->own_by_npy = obj.own_by_npy;
            refcount[this->ptr] += 1;
        }
       return *this;
    };
private:
    static std::map<char*, int> refcount;
    char* ptr;
    bool own_by_npy;
};

typedef struct CArray_RPC_t {
    npy_int32 item_size;
    npy_int32 item_type; // This is a char, use 32bits for alignment.
    npy_int64 size; 
    npy_int32 nd;
    npy_intp dimensions[NPY_MAXDIMS];
    char data[0];
} CArray_RPC;

class CArray {
public:
    CArray(void) : nd(0), type(NPY_INTLTR), type_size(sizeof(npy_int)) {};
    // For this constructor, CArray owns the data and will delete it in the destructor.
    CArray(npy_intp dimensions[], int nd, char type);
    // For this constructor, CArray doesn't owns the data won't delete it, only Py_INCREF 
    // or Py_DECREF the source.
    CArray(npy_intp dimensions[], int nd, char type, char *data, PyArrayObject *source);
    // For this constructor, CArray uses the data and owns the source and will delete
    // it in the destructor. Note that source is a NpyMemManager. This means that it's
    // possible that many objects own the same source. NpyMemManager delete only delete
    // the real data when the reference count is 0.
    CArray(npy_intp dimensions[], int nd, char type, char *data, NpyMemManager *source);
    // The same as previous one except the source is from RPC 
    CArray(CArray_RPC *rpc, NpyMemManager *source);
    ~CArray();

    void copy(char *dest, CExtent *ex); /* dest = this[ex] */

    // This won't call Python APIs
    void to_carray_rpc(CArray_RPC *rpc, CExtent *ex);
    // This will call Python APIs
    PyObject* to_npy(void);

    // Setter and getter
    int get_nd(void) { return nd; };
    char* get_data(void) { return data; };
    npy_intp* get_strides(void) { return strides; };
    npy_intp* get_dimensions(void) { return dimensions; };
    npy_intp get_size(void) { return size; };
    char get_type(void) { return type; };
    int get_type_size(void) { return type_size; };

private:
    void init(npy_intp dimensions[], int nd, char type);
    NpyMemManager *data_source;

    int nd;
    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    npy_intp size;
    char type;
    int type_size;
    char *data;
};



#endif
