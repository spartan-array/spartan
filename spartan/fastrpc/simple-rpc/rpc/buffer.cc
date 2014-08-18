#include <sstream>

#include "buffer.h"

using namespace std;

namespace rpc {

#ifdef RPC_STATISTICS

// -1, 0~15, 16~31, 32~63, 64~127, 128~255, 256~511, 512~1023, 1024~2047, 2048~4095, 4096~8191, 8192~
static Counter g_marshal_in_stat[12];
static Counter g_marshal_in_stat_cumulative[12];
static Counter g_marshal_out_stat[12];
static Counter g_marshal_out_stat_cumulative[12];
static uint64_t g_marshal_stat_report_time = 0;
static const uint64_t g_marshal_stat_report_interval = 1000 * 1000 * 1000;

static void stat_marshal_report() {
    Log::info("* MARSHAL:     -1 0~15 16~31 32~63 64~127 128~255 256~511 512~1023 1024~2047 2048~4095 4096~8191 8192~");
    {
        ostringstream ostr;
        for (size_t i = 0; i < arraysize(g_marshal_in_stat); i++) {
            i64 v = g_marshal_in_stat[i].peek_next();
            g_marshal_in_stat_cumulative[i].next(v);
            ostr << " " << v;
            g_marshal_in_stat[i].reset();
        }
        Log::info("* MARSHAL IN: %s", ostr.str().c_str());
    }
    {
        ostringstream ostr;
        for (size_t i = 0; i < arraysize(g_marshal_in_stat); i++) {
            ostr << " " << g_marshal_in_stat_cumulative[i].peek_next();
        }
        Log::info("* MARSHAL IN (cumulative): %s", ostr.str().c_str());
    }
    {
        ostringstream ostr;
        for (size_t i = 0; i < arraysize(g_marshal_out_stat); i++) {
            i64 v = g_marshal_out_stat[i].peek_next();
            g_marshal_out_stat_cumulative[i].next(v);
            ostr << " " << v;
            g_marshal_out_stat[i].reset();
        }
        Log::info("* MARSHAL OUT:%s", ostr.str().c_str());
    }
    {
        ostringstream ostr;
        for (size_t i = 0; i < arraysize(g_marshal_in_stat); i++) {
            ostr << " " << g_marshal_out_stat_cumulative[i].peek_next();
        }
        Log::info("* MARSHAL OUT (cumulative): %s", ostr.str().c_str());
    }
}

void stat_marshal_in(int fd, const void* buf, size_t nbytes, ssize_t ret) {
    if (ret == -1) {
        g_marshal_in_stat[0].next();
    } else if (ret < 16) {
        g_marshal_in_stat[1].next();
    } else if (ret < 32) {
        g_marshal_in_stat[2].next();
    } else if (ret < 64) {
        g_marshal_in_stat[3].next();
    } else if (ret < 128) {
        g_marshal_in_stat[4].next();
    } else if (ret < 256) {
        g_marshal_in_stat[5].next();
    } else if (ret < 512) {
        g_marshal_in_stat[6].next();
    } else if (ret < 1024) {
        g_marshal_in_stat[7].next();
    } else if (ret < 2048) {
        g_marshal_in_stat[8].next();
    } else if (ret < 4096) {
        g_marshal_in_stat[9].next();
    } else if (ret < 8192) {
        g_marshal_in_stat[10].next();
    } else {
        g_marshal_in_stat[11].next();
    }

    uint64_t now = base::rdtsc();
    if (now - g_marshal_stat_report_time > g_marshal_stat_report_interval) {
        stat_marshal_report();
        g_marshal_stat_report_time = now;
    }
}

void stat_marshal_out(int fd, const void* buf, size_t nbytes, ssize_t ret) {
    if (ret == -1) {
        g_marshal_out_stat[0].next();
    } else if (ret < 16) {
        g_marshal_out_stat[1].next();
    } else if (ret < 32) {
        g_marshal_out_stat[2].next();
    } else if (ret < 64) {
        g_marshal_out_stat[3].next();
    } else if (ret < 128) {
        g_marshal_out_stat[4].next();
    } else if (ret < 256) {
        g_marshal_out_stat[5].next();
    } else if (ret < 512) {
        g_marshal_out_stat[6].next();
    } else if (ret < 1024) {
        g_marshal_out_stat[7].next();
    } else if (ret < 2048) {
        g_marshal_out_stat[8].next();
    } else if (ret < 4096) {
        g_marshal_out_stat[9].next();
    } else if (ret < 8192) {
        g_marshal_out_stat[10].next();
    } else {
        g_marshal_out_stat[11].next();
    }

    uint64_t now = base::rdtsc();
    if (now - g_marshal_stat_report_time > g_marshal_stat_report_interval) {
        stat_marshal_report();
        g_marshal_stat_report_time = now;
    }
}

#endif // RPC_STATISTICS

/**
 * 8kb minimum chunk size.
 * NOTE: this value directly affects how many read/write syscall will be issued.
 */
const size_t raw_bytes::min_size = 8192;


int chunk::write_to_fd(int fd) {
    assert(write_idx <= data->size);
    int cnt = ::write(fd, data->ptr + read_idx, write_idx - read_idx);

#ifdef RPC_STATISTICS
    stat_marshal_out(fd, data->ptr + write_idx, data->size - write_idx, cnt);
#endif // RPC_STATISTICS

    if (cnt > 0) {
        read_idx += cnt;
    }

    assert(write_idx <= data->size);
    return cnt;
}

int chunk::read_from_fd(int fd) {
    assert(write_idx <= data->size);
    assert(read_idx <= write_idx);

    int cnt = 0;
    if (write_idx < data->size) {
        cnt = ::read(fd, data->ptr + write_idx, data->size - write_idx);

#ifdef RPC_STATISTICS
        stat_marshal_in(fd, data->ptr + write_idx, data->size - write_idx, cnt);
#endif // RPC_STATISTICS

        if (cnt > 0) {
            write_idx += cnt;
        }
    }

    assert(write_idx <= data->size);
    assert(read_idx <= write_idx);
    return cnt;
}



UnboundedBuffer::~UnboundedBuffer() {
    chunk* chnk = head_;
    while (chnk != nullptr) {
        chunk* next = chnk->next;
        delete chnk;
        chnk = next;
    }
}

size_t UnboundedBuffer::content_size_slow() const {
    assert(tail_ == nullptr || tail_->next == nullptr);

    size_t sz = 0;
    chunk* chnk = head_;
    while (chnk != nullptr) {
        sz += chnk->content_size();
        chnk = chnk->next;
    }
    return sz;
}

size_t UnboundedBuffer::write(const void* p, size_t n) {
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

size_t UnboundedBuffer::read(void* p, size_t n) {
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

size_t UnboundedBuffer::peek(void* p, size_t n) const {
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

size_t UnboundedBuffer::read_from_fd(int fd) {
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

size_t UnboundedBuffer::read_from_marshal(UnboundedBuffer& m, size_t n) {
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


size_t UnboundedBuffer::write_to_fd(int fd) {
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

bookmark* UnboundedBuffer::set_bookmark(size_t n) {
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

}   // namespace rpc
