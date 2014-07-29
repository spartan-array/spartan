#include <utility>

#include <fcntl.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <netinet/tcp.h>

#include "utils.h"

using namespace std;

namespace rpc {

int set_nonblocking(int fd, bool nonblocking) {
    int ret = fcntl(fd, F_GETFL, 0);
    if (ret != -1) {
        if (nonblocking) {
            ret = fcntl(fd, F_SETFL, ret | O_NONBLOCK);
        } else {
            ret = fcntl(fd, F_SETFL, ret & ~O_NONBLOCK);
        }
    }
    return ret;
}

int open_socket(const char* addr, const struct addrinfo* hints,
                std::function<bool(int, const struct sockaddr*, socklen_t)> filter /* =? */,
                struct sockaddr** p_addr /* =? */, socklen_t* p_len /* =? */) {

    int sock = -1;
    string str_addr(addr);
    size_t idx = str_addr.find(":");
    if (idx == string::npos) {
        Log_error("open_socket(): bad address: %s", addr);
        return -1;
    }
    string host = str_addr.substr(0, idx);
    string port = str_addr.substr(idx + 1);

    struct addrinfo *result, *rp;
    int r = getaddrinfo((host == "0.0.0.0") ? nullptr : host.c_str(), port.c_str(), hints, &result);
    if (r != 0) {
        Log_error("getaddrinfo(): %s", gai_strerror(r));
        return -1;
    }

    for (rp = result; rp != nullptr; rp = rp->ai_next) {
        sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock == -1) {
            continue;
        } else if (filter != nullptr && filter(sock, rp->ai_addr, rp->ai_addrlen) == false) {
            close(sock);
            sock = -1;
            continue;
        } else {
            break;
        }
    }

    if (rp == nullptr) {
        Log_error("open_socket(): failed to open proper socket %s", strerror(errno));
        sock = -1;
    } else if (p_addr != nullptr && p_len != nullptr) {
        *p_addr = (struct sockaddr *) malloc(rp->ai_addrlen);
        *p_len = rp->ai_addrlen;
        memcpy(*p_addr, rp->ai_addr, *p_len);
    }

    freeaddrinfo(result);
    return sock;
}

int tcp_connect(const char* addr) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM; // tcp

    return open_socket(addr, &hints,
                        [] (int sock, const struct sockaddr* sock_addr, socklen_t sock_len) {
                            const int yes = 1;
                            verify(setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) == 0);
                            verify(setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)) == 0);
                            return ::connect(sock, sock_addr, sock_len) == 0;
                        });
}

int udp_connect(const char* addr, struct sockaddr** p_addr /* =? */, socklen_t* p_len /* =? */) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM; // UDP
    hints.ai_protocol = IPPROTO_UDP;
    return open_socket(addr, &hints, nullptr, p_addr, p_len);
}

int udp_bind(const char* addr) {
    // http://web.cecs.pdx.edu/~jrb/tcpip/sockets/ipv6.src/udp/udpclient.c
    struct addrinfo udp_hints;
    memset(&udp_hints, 0, sizeof(struct addrinfo));
    udp_hints.ai_family = AF_INET;
    udp_hints.ai_socktype = SOCK_DGRAM; // udp
    udp_hints.ai_protocol = IPPROTO_UDP;

    return open_socket(addr, &udp_hints,
                        [] (int sock, const struct sockaddr* sock_addr, socklen_t sock_len) {
                            return ::bind(sock, sock_addr, sock_len) == 0;
                        });
}

}
