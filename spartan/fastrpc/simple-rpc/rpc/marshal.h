#pragma once

#include <list>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <limits>

#include <inttypes.h>
#include <string.h>

#include "utils.h"
#include "buffer.h"

namespace rpc {


// not thread safe, for better performance
class Marshal: public NoCopy {
    chunk* head_;
    chunk* tail_;
    i32 write_cnt_;
    size_t content_size_;

    // for debugging purpose
    size_t content_size_slow() const;

public:

    Marshal(): head_(nullptr), tail_(nullptr), write_cnt_(0), content_size_(0) { }
    ~Marshal();

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
    size_t read_from_marshal(Marshal& m, size_t n);

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


inline rpc::Marshal& operator <<(rpc::Marshal& m, const rpc::i8& v) {
    verify(m.write(&v, sizeof(v)) == sizeof(v));
    return m;
}

inline rpc::Marshal& operator <<(rpc::Marshal& m, const rpc::i16& v) {
    verify(m.write(&v, sizeof(v)) == sizeof(v));
    return m;
}

inline rpc::Marshal& operator <<(rpc::Marshal& m, const rpc::i32& v) {
    verify(m.write(&v, sizeof(v)) == sizeof(v));
    return m;
}

inline rpc::Marshal& operator <<(rpc::Marshal& m, const rpc::i64& v) {
    verify(m.write(&v, sizeof(v)) == sizeof(v));
    return m;
}

inline rpc::Marshal& operator <<(rpc::Marshal& m, const rpc::v32& v) {
    char buf[5];
    size_t bsize = base::SparseInt::dump(v.get(), buf);
    verify(m.write(buf, bsize) == bsize);
    return m;
}

inline rpc::Marshal& operator <<(rpc::Marshal& m, const rpc::v64& v) {
    char buf[9];
    size_t bsize = base::SparseInt::dump(v.get(), buf);
    verify(m.write(buf, bsize) == bsize);
    return m;
}

inline rpc::Marshal& operator <<(rpc::Marshal& m, const uint8_t& u) {
    verify(m.write(&u, sizeof(u)) == sizeof(u));
    return m;
}

inline rpc::Marshal& operator <<(rpc::Marshal& m, const uint16_t& u) {
    verify(m.write(&u, sizeof(u)) == sizeof(u));
    return m;
}

inline rpc::Marshal& operator <<(rpc::Marshal& m, const uint32_t& u) {
    verify(m.write(&u, sizeof(u)) == sizeof(u));
    return m;
}

inline rpc::Marshal& operator <<(rpc::Marshal& m, const uint64_t& u) {
    verify(m.write(&u, sizeof(u)) == sizeof(u));
    return m;
}

inline rpc::Marshal& operator <<(rpc::Marshal& m, const double& v) {
    verify(m.write(&v, sizeof(v)) == sizeof(v));
    return m;
}

inline rpc::Marshal& operator <<(rpc::Marshal& m, const std::string& v) {
    v64 v_len = v.length();
    m << v_len;
    if (v_len.get() > 0) {
        verify(m.write(v.c_str(), v_len.get()) == (size_t) v_len.get());
    }
    return m;
}

template<class T1, class T2>
inline rpc::Marshal& operator <<(rpc::Marshal& m, const std::pair<T1, T2>& v) {
    m << v.first;
    m << v.second;
    return m;
}

template<class T>
inline rpc::Marshal& operator <<(rpc::Marshal& m, const std::vector<T>& v) {
    v64 v_len = v.size();
    m << v_len;
    for (typename std::vector<T>::const_iterator it = v.begin(); it != v.end(); ++it) {
        m << *it;
    }
    return m;
}

template<class T>
inline rpc::Marshal& operator <<(rpc::Marshal& m, const std::list<T>& v) {
    v64 v_len = v.size();
    m << v_len;
    for (typename std::list<T>::const_iterator it = v.begin(); it != v.end(); ++it) {
        m << *it;
    }
    return m;
}

template<class T>
inline rpc::Marshal& operator <<(rpc::Marshal& m, const std::set<T>& v) {
    v64 v_len = v.size();
    m << v_len;
    for (typename std::set<T>::const_iterator it = v.begin(); it != v.end(); ++it) {
        m << *it;
    }
    return m;
}

template<class K, class V>
inline rpc::Marshal& operator <<(rpc::Marshal& m, const std::map<K, V>& v) {
    v64 v_len = v.size();
    m << v_len;
    for (typename std::map<K, V>::const_iterator it = v.begin(); it != v.end(); ++it) {
        m << it->first << it->second;
    }
    return m;
}

template<class T>
inline rpc::Marshal& operator <<(rpc::Marshal& m, const std::unordered_set<T>& v) {
    v64 v_len = v.size();
    m << v_len;
    for (typename std::unordered_set<T>::const_iterator it = v.begin(); it != v.end(); ++it) {
        m << *it;
    }
    return m;
}

template<class K, class V>
inline rpc::Marshal& operator <<(rpc::Marshal& m, const std::unordered_map<K, V>& v) {
    v64 v_len = v.size();
    m << v_len;
    for (typename std::unordered_map<K, V>::const_iterator it = v.begin(); it != v.end(); ++it) {
        m << it->first << it->second;
    }
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, rpc::i8& v) {
    verify(m.read(&v, sizeof(v)) == sizeof(v));
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, rpc::i16& v) {
    verify(m.read(&v, sizeof(v)) == sizeof(v));
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, rpc::i32& v) {
    verify(m.read(&v, sizeof(v)) == sizeof(v));
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, rpc::i64& v) {
    verify(m.read(&v, sizeof(v)) == sizeof(v));
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, rpc::v32& v) {
    char byte0;
    verify(m.peek(&byte0, 1) == 1);
    size_t bsize = base::SparseInt::buf_size(byte0);
    char buf[5];
    verify(m.read(buf, bsize) == bsize);
    i32 val = base::SparseInt::load_i32(buf);
    v.set(val);
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, rpc::v64& v) {
    char byte0;
    verify(m.peek(&byte0, 1) == 1);
    size_t bsize = base::SparseInt::buf_size(byte0);
    char buf[9];
    verify(m.read(buf, bsize) == bsize);
    i64 val = base::SparseInt::load_i64(buf);
    v.set(val);
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, uint8_t& u) {
    verify(m.read(&u, sizeof(u)) == sizeof(u));
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, uint16_t& u) {
    verify(m.read(&u, sizeof(u)) == sizeof(u));
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, uint32_t& u) {
    verify(m.read(&u, sizeof(u)) == sizeof(u));
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, uint64_t& u) {
    verify(m.read(&u, sizeof(u)) == sizeof(u));
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, double& v) {
    verify(m.read(&v, sizeof(v)) == sizeof(v));
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, std::string& v) {
    v64 v_len;
    m >> v_len;
    v.resize(v_len.get());
    if (v_len.get() > 0) {
        verify(m.read(&v[0], v_len.get()) == (size_t) v_len.get());
    }
    return m;
}

template<class T1, class T2>
inline rpc::Marshal& operator >>(rpc::Marshal& m, std::pair<T1, T2>& v) {
    m >> v.first;
    m >> v.second;
    return m;
}

template<class T>
inline rpc::Marshal& operator >>(rpc::Marshal& m, std::vector<T>& v) {
    v64 v_len;
    m >> v_len;
    v.clear();
    v.reserve(v_len.get());
    for (int i = 0; i < v_len.get(); i++) {
        T elem;
        m >> elem;
        v.push_back(elem);
    }
    return m;
}

template<class T>
inline rpc::Marshal& operator >>(rpc::Marshal& m, std::list<T>& v) {
    v64 v_len;
    m >> v_len;
    v.clear();
    for (int i = 0; i < v_len.get(); i++) {
        T elem;
        m >> elem;
        v.push_back(elem);
    }
    return m;
}

template<class T>
inline rpc::Marshal& operator >>(rpc::Marshal& m, std::set<T>& v) {
    v64 v_len;
    m >> v_len;
    v.clear();
    for (int i = 0; i < v_len.get(); i++) {
        T elem;
        m >> elem;
        v.insert(elem);
    }
    return m;
}

template<class K, class V>
inline rpc::Marshal& operator >>(rpc::Marshal& m, std::map<K, V>& v) {
    v64 v_len;
    m >> v_len;
    v.clear();
    for (int i = 0; i < v_len.get(); i++) {
        K key;
        V value;
        m >> key >> value;
        insert_into_map(v, key, value);
    }
    return m;
}

template<class T>
inline rpc::Marshal& operator >>(rpc::Marshal& m, std::unordered_set<T>& v) {
    v64 v_len;
    m >> v_len;
    v.clear();
    for (int i = 0; i < v_len.get(); i++) {
        T elem;
        m >> elem;
        v.insert(elem);
    }
    return m;
}

template<class K, class V>
inline rpc::Marshal& operator >>(rpc::Marshal& m, std::unordered_map<K, V>& v) {
    v64 v_len;
    m >> v_len;
    v.clear();
    for (int i = 0; i < v_len.get(); i++) {
        K key;
        V value;
        m >> key >> value;
        insert_into_map(v, key, value);
    }
    return m;
}


class UdpBuffer {
    Marshal m_;
    char* buf_;

public:
    static const size_t max_udp_packet_size_s = 65507;

    UdpBuffer() {
        buf_ = new char[max_udp_packet_size_s];
    }
    ~UdpBuffer() {
        delete[] buf_;
    }
    char* get_buf(size_t* size, bool* overflow) {
        *overflow = false;
        *size = 0;
        if (!m_.empty()) {
            if (m_.content_size() > max_udp_packet_size_s) {
                *overflow = true;
                *size = max_udp_packet_size_s;
                return nullptr;
            } else {
                *size = m_.read(buf_, m_.content_size());
            }
        }
        return buf_;
    }
    Marshal& base() {
        return m_;
    }
};


// used only in Python extension
inline UdpBuffer& operator<< (UdpBuffer& udp, Marshal& v) {
    udp.base().read_from_marshal(v, v.content_size());
    return udp;
}

template <class T>
inline UdpBuffer& operator<< (UdpBuffer& udp, const T& v) {
    udp.base() << v;
    return udp;
}

template <class T>
inline UdpBuffer& operator>> (UdpBuffer& udp, T& v) {
    udp.base() >> v;
    return udp;
}


} // namespace rpc
