#include <sys/time.h>

#include "basetypes.h"

namespace base {


size_t SparseInt::buf_size(char byte0) {
    if ((byte0 & 0x80) == 0) {
        // binary: 0...
        return 1;
    } else if ((byte0 & 0xC0) == 0x80) {
        // binary: 10...
        return 2;
    } else if ((byte0 & 0xE0) == 0xC0) {
        // binary: 110...
        return 3;
    } else if ((byte0 & 0xF0) == 0xE0) {
        // binary: 1110...
        return 4;
    } else if ((byte0 & 0xF8) == 0xF0) {
        // binary: 11110...
        return 5;
    } else if ((byte0 & 0xFC) == 0xF8) {
        // binary: 111110...
        return 6;
    } else if ((byte0 & 0xFE) == 0xFC) {
        // binary: 1111110...
        return 7;
    } else if ((byte0 & 0xFF) == 0xFE) {
        // binary: 11111110...
        return 8;
    } else {
        return 9;
    }
}

size_t SparseInt::val_size(i64 val) {
    if (-64 <= val && val <= 63) {
        return 1;
    } else if (-8192 <= val && val <= 8191) {
        return 2;
    } else if (-1048576 <= val && val <= 1048575) {
        return 3;
    } else if (-134217728 <= val && val <= 134217727) {
        return 4;
    } else if (-17179869184LL <= val && val <= 17179869183LL) {
        return 5;
    } else if (-2199023255552LL <= val && val <= 2199023255551LL) {
        return 6;
    } else if (-281474976710656LL <= val && val <= 281474976710655LL) {
        return 7;
    } else if (-36028797018963968LL <= val && val <= 36028797018963967LL) {
        return 8;
    } else {
        return 9;
    }
}

size_t SparseInt::dump(i32 val, char* buf) {
    char* pv = (char *) &val;
    if (-64 <= val && val <= 63) {
        buf[0] = pv[0];
        buf[0] &= 0x7F;
        return 1;
    } else if (-8192 <= val && val <= 8191) {
        buf[0] = pv[1];
        buf[1] = pv[0];
        buf[0] &= 0x3F;
        buf[0] |= 0x80;
        return 2;
    } else if (-1048576 <= val && val <= 1048575) {
        buf[0] = pv[2];
        buf[1] = pv[1];
        buf[2] = pv[0];
        buf[0] &= 0x1F;
        buf[0] |= 0xC0;
        return 3;
    } else if (-134217728 <= val && val <= 134217727) {
        buf[0] = pv[3];
        buf[1] = pv[2];
        buf[2] = pv[1];
        buf[3] = pv[0];
        buf[0] &= 0x0F;
        buf[0] |= 0xE0;
        return 4;
    } else {
        buf[1] = pv[3];
        buf[2] = pv[2];
        buf[3] = pv[1];
        buf[4] = pv[0];
        if (val < 0) {
            buf[0] = 0xF7;
        } else {
            buf[0] = 0xF0;
        }
        return 5;
    }
}

size_t SparseInt::dump(i64 val, char* buf) {
    char* pv = (char *) &val;
    if (-64 <= val && val <= 63) {
        buf[0] = pv[0];
        buf[0] &= 0x7F;
        return 1;
    } else if (-8192 <= val && val <= 8191) {
        buf[0] = pv[1];
        buf[1] = pv[0];
        buf[0] &= 0x3F;
        buf[0] |= 0x80;
        return 2;
    } else if (-1048576 <= val && val <= 1048575) {
        buf[0] = pv[2];
        buf[1] = pv[1];
        buf[2] = pv[0];
        buf[0] &= 0x1F;
        buf[0] |= 0xC0;
        return 3;
    } else if (-134217728 <= val && val <= 134217727) {
        buf[0] = pv[3];
        buf[1] = pv[2];
        buf[2] = pv[1];
        buf[3] = pv[0];
        buf[0] &= 0x0F;
        buf[0] |= 0xE0;
        return 4;
    } else if (-17179869184LL <= val && val <= 17179869183LL) {
        buf[0] = pv[4];
        buf[1] = pv[3];
        buf[2] = pv[2];
        buf[3] = pv[1];
        buf[4] = pv[0];
        buf[0] &= 0x07;
        buf[0] |= 0xF0;
        return 5;
    } else if (-2199023255552LL <= val && val <= 2199023255551LL) {
        buf[0] = pv[5];
        buf[1] = pv[4];
        buf[2] = pv[3];
        buf[3] = pv[2];
        buf[4] = pv[1];
        buf[5] = pv[0];
        buf[0] &= 0x03;
        buf[0] |= 0xF8;
        return 6;
    } else if (-281474976710656LL <= val && val <= 281474976710655LL) {
        buf[0] = pv[6];
        buf[1] = pv[5];
        buf[2] = pv[4];
        buf[3] = pv[3];
        buf[4] = pv[2];
        buf[5] = pv[1];
        buf[6] = pv[0];
        buf[0] &= 0x01;
        buf[0] |= 0xFC;
        return 7;
    } else if (-36028797018963968LL <= val && val <= 36028797018963967LL) {
        buf[1] = pv[7];
        buf[2] = pv[6];
        buf[3] = pv[5];
        buf[4] = pv[4];
        buf[5] = pv[3];
        buf[6] = pv[2];
        buf[7] = pv[1];
        buf[8] = pv[0];
        buf[0] = 0xFE;
        return 8;
    } else {
        buf[1] = pv[7];
        buf[2] = pv[6];
        buf[3] = pv[5];
        buf[4] = pv[4];
        buf[5] = pv[3];
        buf[6] = pv[2];
        buf[7] = pv[1];
        buf[8] = pv[0];
        buf[0] = 0xFF;
        return 9;
    }
}


i32 SparseInt::load_i32(const char* buf) {
    i32 val = 0;
    char* pv = (char *) &val;
    int bsize = SparseInt::buf_size(buf[0]);
    if (bsize < 5) {
        for (int i = 0; i < bsize; i++) {
            pv[i] = buf[bsize - i - 1];
        }
        pv[bsize - 1] &= 0xFF >> bsize;
        if ((pv[bsize - 1] >> (7 - bsize)) & 0x1) {
            pv[bsize - 1] |= 0xFF << (7 - bsize);
            for (int i = bsize; i < 4; i++) {
                pv[i] = 0xFF;
            }
        }
    } else {
        for (int i = 0; i < 4; i++) {
            pv[i] = buf[4 - i];
        }
    }
    return val;
}

i64 SparseInt::load_i64(const char* buf) {
    i64 val = 0;
    char* pv = (char *) &val;
    int bsize = SparseInt::buf_size(buf[0]);
    if (bsize < 8) {
        for (int i = 0; i < bsize; i++) {
            pv[i] = buf[bsize - i - 1];
        }
        pv[bsize - 1] &= 0xFF >> bsize;
        if ((pv[bsize - 1] >> (7 - bsize)) & 0x1) {
            pv[bsize - 1] |= 0xFF << (7 - bsize);
            for (int i = bsize; i < 8; i++) {
                pv[i] = 0xFF;
            }
        }
    } else {
        for (int i = 0; i < 8; i++) {
            pv[i] = buf[8 - i];
        }
    }
    return val;
}


void Timer::start() {
    reset();
    gettimeofday(&begin_, nullptr);
}

void Timer::stop() {
    gettimeofday(&end_, nullptr);
}

void Timer::reset() {
    begin_.tv_sec = 0;
    begin_.tv_usec = 0;
    end_.tv_sec = 0;
    end_.tv_usec = 0;
}

double Timer::elapsed() const {
    verify(begin_.tv_sec != 0 || begin_.tv_usec != 0);
    if (end_.tv_sec == 0 && end_.tv_usec == 0) {
        // not stopped yet
        struct timeval now;
        gettimeofday(&now, nullptr);
        return now.tv_sec - begin_.tv_sec + (now.tv_usec - begin_.tv_usec) / 1000000.0;
    }
    return end_.tv_sec - begin_.tv_sec + (end_.tv_usec - begin_.tv_usec) / 1000000.0;
}

Rand::Rand() {
    struct timeval now;
    gettimeofday(&now, nullptr);
    rand_.seed(now.tv_sec + now.tv_usec + (long long) pthread_self() + (long long) this);
}

} // namespace base
