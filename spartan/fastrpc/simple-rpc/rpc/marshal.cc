#include <sys/time.h>

#include "marshal.h"

using namespace std;

namespace rpc {

Marshal::~Marshal() {
    chunk* chnk = head_;
    while (chnk != nullptr) {
        chunk* next = chnk->next;
        delete chnk;
        chnk = next;
    }
}

size_t Marshal::content_size_slow() const {
    assert(tail_ == nullptr || tail_->next == nullptr);

    size_t sz = 0;
    chunk* chnk = head_;
    while (chnk != nullptr) {
        sz += chnk->content_size();
        chnk = chnk->next;
    }
    return sz;
}

size_t Marshal::write(const void* p, size_t n) {
    assert(tail_ == nullptr || tail_->next == nullptr);

    if (head_ == nullptr) {
        assert(tail_ == nullptr);
        head_ = new chunk(p, n);
        tail_ = head_;
    } else if (tail_->fully_written()) {
        tail_->next = new chunk(p, n);
        tail_ = tail_->next;
    } else {
        size_t n_write = tail_->write(p, n);

        // otherwise the above fully_written() should have returned true
        assert(n_write > 0);

        if (n_write < n) {
            const char* pc = (const char *) p;
            tail_->next = new chunk(pc + n_write, n - n_write);
            tail_ = tail_->next;
        }
    }
    write_cnt_ += n;
    content_size_ += n;
    assert(content_size_ == content_size_slow());

    return n;
}

size_t Marshal::read(void* p, size_t n) {
    assert(tail_ == nullptr || tail_->next == nullptr);
    assert(empty() || (head_ != nullptr && !head_->fully_read()));

    char* pc = (char *) p;
    size_t n_read = 0;
    while (n_read < n && head_ != nullptr && head_->content_size() > 0) {
        size_t cnt = head_->read(pc + n_read, n - n_read);
        if (head_->fully_read()) {
            if (tail_ == head_) {
                // deleted the only chunk
                tail_ = nullptr;
            }
            chunk* chnk = head_;
            head_ = head_->next;
            delete chnk;
        }
        if (cnt == 0) {
            // currently there's no data available, so stop
            break;
        }
        n_read += cnt;
    }
    assert(content_size_ >= n_read);
    content_size_ -= n_read;
    assert(content_size_ == content_size_slow());

    assert(n_read <= n);
    assert(tail_ == nullptr || tail_->next == nullptr);
    assert(empty() || (head_ != nullptr && !head_->fully_read()));

    return n_read;
}

size_t Marshal::peek(void* p, size_t n) const {
    assert(tail_ == nullptr || tail_->next == nullptr);
    assert(empty() || (head_ != nullptr && !head_->fully_read()));

    char* pc = (char *) p;
    size_t n_peek = 0;
    chunk* chnk = head_;
    while (chnk != nullptr && n - n_peek > 0) {
        size_t cnt = chnk->peek(pc + n_peek, n - n_peek);
        if (cnt == 0) {
            // no more data to peek, quit
            break;
        }
        n_peek += cnt;
        chnk = chnk->next;
    }

    assert(n_peek <= n);
    assert(tail_ == nullptr || tail_->next == nullptr);
    assert(empty() || (head_ != nullptr && !head_->fully_read()));
    return n_peek;
}

size_t Marshal::read_from_fd(int fd) {
    assert(empty() || (head_ != nullptr && !head_->fully_read()));

    size_t n_bytes = 0;
    for (;;) {
        if (head_ == nullptr) {
            head_ = new chunk;
            tail_ = head_;
        } else if (tail_->fully_written()) {
            tail_->next = new chunk;
            tail_ = tail_->next;
        }
        int r = tail_->read_from_fd(fd);
        if (r <= 0) {
            break;
        }
        n_bytes += r;
    }
    write_cnt_ += n_bytes;
    content_size_ += n_bytes;
    assert(content_size_ == content_size_slow());

    assert(empty() || (head_ != nullptr && !head_->fully_read()));
    return n_bytes;
}

size_t Marshal::read_from_marshal(Marshal& m, size_t n) {
    assert(m.content_size() >= n);   // require m.content_size() >= n > 0
    size_t n_fetch = 0;

    if ((head_ == nullptr && tail_ == nullptr) || tail_->fully_written()) {
        // efficiently copy data by only copying pointers
        while (n_fetch < n) {
            // NOTE: The copied chunk is shared by 2 Marshal objects. Be careful
            //       that only one Marshal should be able to write to it! For the
            //       given 2 use cases, it works.
            chunk* chnk = m.head_->shared_copy();
            if (n_fetch + chnk->content_size() > n) {
                // only fetch enough bytes we need
                chnk->write_idx -= (n_fetch + chnk->content_size()) - n;
            }
            size_t cnt = chnk->content_size();
            assert(cnt > 0);
            n_fetch += cnt;
            verify(m.head_->discard(cnt) == cnt);
            if (head_ == nullptr) {
                head_ = tail_ = chnk;
            } else {
                tail_->next = chnk;
                tail_ = chnk;
            }
            if (m.head_->fully_read()) {
                if (m.tail_ == m.head_) {
                    // deleted the only chunk
                    m.tail_ = nullptr;
                }
                chunk* next = m.head_->next;
                delete m.head_;
                m.head_ = next;
            }
        }
        write_cnt_ += n_fetch;
        content_size_ += n_fetch;
        verify(m.content_size_ >= n_fetch);
        m.content_size_ -= n_fetch;

    } else {

        // number of bytes that need to be copied
        size_t copy_n = std::min(tail_->data->size - tail_->write_idx, n);
        char* buf = new char[copy_n];
        n_fetch = m.read(buf, copy_n);
        verify(n_fetch == copy_n);
        verify(this->write(buf, n_fetch) == n_fetch);
        delete[] buf;

        size_t leftover = n - copy_n;
        if (leftover > 0) {
            verify(tail_->fully_written());
            n_fetch += this->read_from_marshal(m, leftover);
        }
    }
    assert(n_fetch == n);
    assert(content_size_ == content_size_slow());
    return n_fetch;
}


size_t Marshal::write_to_fd(int fd) {
    size_t n_write = 0;
    while (!empty()) {
        int cnt = head_->write_to_fd(fd);
        if (head_->fully_read()) {
            if (head_ == tail_) {
                tail_ = nullptr;
            }
            chunk* chnk = head_;
            head_ = head_->next;
            delete chnk;
        }
        if (cnt <= 0) {
            // currently there's no data available, so stop
            break;
        }
        assert(content_size_ >= (size_t) cnt);
        content_size_ -= cnt;
        n_write += cnt;
    }
    assert(content_size_ == content_size_slow());
    return n_write;
}

bookmark* Marshal::set_bookmark(size_t n) {
    verify(write_cnt_ == 0);

    bookmark* bm = new bookmark;
    bm->size = n;
    bm->ptr = new char*[bm->size];
    for (size_t i = 0; i < n; i++) {
        if (head_ == nullptr) {
            head_ = new chunk;
            tail_ = head_;
        } else if (tail_->fully_written()) {
            tail_->next = new chunk;
            tail_ = tail_->next;
        }
        bm->ptr[i] = tail_->set_bookmark();
    }
    content_size_ += n;
    assert(content_size_ == content_size_slow());

    return bm;
}

} // namespace rpc
