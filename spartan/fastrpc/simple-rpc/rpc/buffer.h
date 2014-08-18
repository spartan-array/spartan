#pragma once

#include <unistd.h>

#include "utils.h"

namespace rpc {

class Buffer {
public:
    virtual ~Buffer() {}
    virtual size_t content_size() const = 0;
    virtual size_t write(const void* p, size_t n) = 0;
    virtual size_t read(void* p, size_t n) = 0;
    virtual size_t peek(void* p, size_t n) const = 0;
};


struct raw_bytes: public RefCounted {
    char* ptr;
    size_t size;
    static const size_t min_size;

    raw_bytes(size_t sz = min_size) {
        size = std::max(sz, min_size);
        ptr = new char[size];
    }
    raw_bytes(const void* p, size_t n) {
        size = std::max(n, min_size);
        ptr = new char[size];
        memcpy(ptr, p, n);
    }
    ~raw_bytes() { delete[] ptr; }
};


struct bookmark: public NoCopy {
    size_t size;
    char** ptr;

    ~bookmark() {
        delete[] ptr;
    }
};


struct chunk: public NoCopy {
    friend class UdpBuffer;

private:
    chunk(raw_bytes* dt, size_t rd_idx, size_t wr_idx)
            : data((raw_bytes *) dt->ref_copy()), read_idx(rd_idx), write_idx(wr_idx), next(nullptr) {
        assert(write_idx <= data->size);
        assert(read_idx <= write_idx);
    }

public:

    raw_bytes* data;
    size_t read_idx;
    size_t write_idx;
    chunk* next;

    chunk(): data(new raw_bytes), read_idx(0), write_idx(0), next(nullptr) {}
    chunk(const void* p, size_t n): data(new raw_bytes(p, n)), read_idx(0), write_idx(n), next(nullptr) {}
    ~chunk() { data->release(); }

    // NOTE: This function is only intended for Marshal::read_from_marshal.
    chunk* shared_copy() const {
        return new chunk(data, read_idx, write_idx);
    }

    size_t content_size() const {
        assert(write_idx <= data->size);
        assert(read_idx <= write_idx);
        return write_idx - read_idx;
    }

    char* set_bookmark() {
        assert(write_idx <= data->size);
        assert(read_idx <= write_idx);

        char* p = &data->ptr[write_idx];
        write_idx++;

        assert(write_idx <= data->size);
        assert(read_idx <= write_idx);
        return p;
    }

    size_t write(const void* p, size_t n) {
        assert(write_idx <= data->size);
        assert(read_idx <= write_idx);

        size_t n_write = std::min(n, data->size - write_idx);
        if (n_write > 0) {
            memcpy(data->ptr + write_idx, p, n_write);
        }
        write_idx += n_write;

        assert(write_idx <= data->size);
        assert(read_idx <= write_idx);
        return n_write;
    }

    size_t read(void* p, size_t n) {
        assert(write_idx <= data->size);
        assert(read_idx <= write_idx);

        size_t n_read = std::min(n, write_idx - read_idx);
        if (n_read > 0) {
            memcpy(p, data->ptr + read_idx, n_read);
        }
        read_idx += n_read;

        assert(write_idx <= data->size);
        assert(read_idx <= write_idx);
        return n_read;
    }

    size_t peek(void* p, size_t n) const {
        assert(write_idx <= data->size);
        assert(read_idx <= write_idx);

        size_t n_peek = std::min(n, write_idx - read_idx);
        if (n_peek > 0) {
            memcpy(p, data->ptr + read_idx, n_peek);
        }

        return n_peek;
    }

    size_t discard(size_t n) {
        assert(write_idx <= data->size);
        assert(read_idx <= write_idx);

        size_t n_discard = std::min(n, write_idx - read_idx);
        read_idx += n_discard;

        assert(write_idx <= data->size);
        assert(read_idx <= write_idx);
        return n_discard;
    }

    int write_to_fd(int fd);
    int read_from_fd(int fd);

    // check if it is not possible to write to the chunk anymore.
    bool fully_written() const {
        assert(write_idx <= data->size);
        assert(read_idx <= write_idx);
        return write_idx == data->size;
    }

    // check if it is not possible to read any data even if retry later
    bool fully_read() const {
        assert(write_idx <= data->size);
        assert(read_idx <= write_idx);
        return read_idx == data->size;
    }
};


// not thread safe, for better performance
class UnboundedBuffer: public Buffer {
    MAKE_NOCOPY(UnboundedBuffer);

    chunk* head_;
    chunk* tail_;
    i32 write_cnt_;
    size_t content_size_;

    // for debugging purpose
    size_t content_size_slow() const;

public:

    UnboundedBuffer(): head_(nullptr), tail_(nullptr), write_cnt_(0), content_size_(0) { }
    ~UnboundedBuffer();

    bool empty() const {
        assert(content_size_ == content_size_slow());
        return content_size_ == 0;
    }
    size_t content_size() const {
        assert(content_size_ == content_size_slow());
        return content_size_;
    }

    size_t write(const void* p, size_t n);
    size_t read(void* p, size_t n);
    size_t peek(void* p, size_t n) const;

    size_t read_from_fd(int fd);

    // NOTE: This function is only used *internally* to chop a slice of marshal object.
    // Use case 1: In C++ server io thread, when a compelete packet is received, read it off
    //             into a Marshal object and hand over to worker threads.
    // Use case 2: In Python extension, buffer message in Marshal object, and send to network.
    size_t read_from_marshal(UnboundedBuffer& m, size_t n);

    size_t write_to_fd(int fd);

    bookmark* set_bookmark(size_t n);
    void write_bookmark(bookmark* bm, const void* p) {
        const char* pc = (const char *) p;
        assert(bm != nullptr && bm->ptr != nullptr && p != nullptr);
        for (size_t i = 0; i < bm->size; i++) {
            *(bm->ptr[i]) = pc[i];
        }
    }

    i32 get_and_reset_write_cnt() {
        i32 cnt = write_cnt_;
        write_cnt_ = 0;
        return cnt;
    }
};



}   // namespace rpc
