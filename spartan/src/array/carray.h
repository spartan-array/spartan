#ifndef __CARRAY_H__
#define __CARRAY_H__
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include "cextent.h"
#include "rpc/marshal.h"

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
/**
 * TODO: Is this class has to be locked ?
 */
class NpyMemManager {
public:
    NpyMemManager(void) : size(0), source(NULL), data(NULL), own_by_npy(false){};
    NpyMemManager(char* source, char *data, bool own_by_npy, unsigned long size) {
        std::map<char*, int>::iterator it;

        this->source = source;
        this->data = data;
        this->own_by_npy = own_by_npy;
        this->size = size;
        if ((it = NpyMemManager::refcount.find(source)) == NpyMemManager::refcount.end()) {
            refcount[source] = 1; 
            if (own_by_npy) {
                Py_INCREF(source);
            }
        } else {
            it->second += 1;
        }
    }

    NpyMemManager(const NpyMemManager& mgr) {
        source = mgr.source;
        data = mgr.data;
        own_by_npy = mgr.own_by_npy;
        size = mgr.size;
        refcount[source] += 1;
    }

    ~NpyMemManager(void) {
        auto it = NpyMemManager::refcount.find(source);
        if (it != NpyMemManager::refcount.end()) {
            if (it->second == 1) {
                refcount.erase(source);
                if (own_by_npy) {
                    Py_DECREF(source);
                } else {
                    delete source;
                }
            } else {
                it->second -= 1;
            }
        }
    }

    // A dangerous function which should only be used when you actually know what this is for.
    void clear() {
        auto it = NpyMemManager::refcount.find(source);
        if (it != NpyMemManager::refcount.end()) {
            refcount.erase(source);
        }
    }

    int get_refcount(void) {
        auto it = NpyMemManager::refcount.find(source);
        if (it != NpyMemManager::refcount.end()) {
            return it->second;
        } else {
            return -1;
        }
    }

    char* get_source() { return source; }
    char* get_data() { return data; }

    unsigned long size;
private:
    NpyMemManager& operator=(const NpyMemManager& obj) = delete;
    static std::map<char*, int> refcount;
    char* source;
    char* data;
    bool own_by_npy;
};

typedef struct CArray_RPC_t {
    npy_int32 item_size;
    npy_int32 item_type; // This is a char, use 32bits for alignment.
    npy_int64 size; 
    npy_int32 nd;
    npy_intp dimensions[NPY_MAXDIMS];
    bool is_npy_memmanager;
    char *data;
} CArray_RPC;

class CArray {
public:
    CArray(void) : nd(0), type(NPY_INTLTR), type_size(sizeof(npy_int)) {};
    // For this constructor, CArray owns the data and will delete it in the destructor.
    CArray(npy_intp dimensions[], int nd, char type);
    // For this constructor, CArray owns the data and will delete it in the destructor.
    CArray(npy_intp dimensions[], int nd, char type, char *buffer);
    // For this constructor, CArray doesn't owns the data and won't delete it, only 
    // Py_INCREF or Py_DECREF the source.
    CArray(npy_intp dimensions[], int nd, char type, char *data, PyArrayObject *source);
    // For this constructor, CArray uses the data and owns the source and will delete
    // it in the destructor. Note that source is a NpyMemManager. This means that it's
    // possible that many objects own the same source. NpyMemManager delete only delete
    // the real data when the reference count is 0.
    CArray(npy_intp dimensions[], int nd, char type, char *data, NpyMemManager *source);
    // The source is from RPC and the ownership is passed to this CArray
    CArray(CArray_RPC *rpc);
    ~CArray();

    npy_intp copy_slice(CExtent *ex, NpyMemManager **dest);

    // This won't call Python APIs
    std::vector <char*> to_carray_rpc(CExtent *ex);
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

    // For RPC marshal
    friend rpc::Marshal& operator <<(rpc::Marshal&, const CArray&);
    friend rpc::Marshal& operator >>(rpc::Marshal&, CArray& o) ;

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


inline rpc::Marshal& operator <<(rpc::Marshal& m, const CArray& o) 
{
    npy_intp dimensions[NPY_MAXDIMS];

    m << o.nd;
    m.write(&(o.type), sizeof(o.type));
    m.write(dimensions, sizeof(npy_intp) * NPY_MAXDIMS);
    m.write(o.data, o.size);
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal&m, CArray& o) 
{
    npy_intp dimensions[NPY_MAXDIMS];
    int nd;
    char type;

    m >> nd;
    m.read(&type, sizeof(char));
    m.read(dimensions, sizeof(npy_intp) * NPY_MAXDIMS);
    o.init(dimensions, nd, type);
    o.data = (char*) malloc(o.size);
    assert(o.data != NULL);
    o.data_source = new NpyMemManager(o.data, o.data, false, o.size);
    assert(m.content_size() == o.size);
    m.read(o.data, o.size);
    return m;
}
#endif
