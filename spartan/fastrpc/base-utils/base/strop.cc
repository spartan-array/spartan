#include <iostream>
#include <sstream>
#include <string.h>

#include "strop.h"

namespace base {

bool startswith(const char* str, const char* head) {
    size_t len_str = strlen(str);
    size_t len_head = strlen(head);
    if (len_head > len_str) {
        return false;
    }
    return strncmp(str, head, len_head) == 0;
}

bool endswith(const char* str, const char* tail) {
    size_t len_str = strlen(str);
    size_t len_tail = strlen(tail);
    if (len_tail > len_str) {
        return false;
    }
    return strncmp(str + (len_str - len_tail), tail, len_tail) == 0;
}

std::string format_decimal(double val) {
    std::ostringstream o;
    o.precision(2);
    o << std::fixed << val;
    std::string s(o.str());
    std::string str;
    size_t idx = 0;
    while (idx < s.size()) {
        if (s[idx] == '.') {
            break;
        }
        idx++;
    }
    str.reserve(s.size() + 16);
    for (size_t i = 0; i < idx; i++) {
        if ((idx - i) % 3 == 0 && i != 0 && s[i - 1] != '-') {
            str += ',';
        }
        str += s[i];
    }
    str += s.substr(idx);
    if (str == "-0.00") {
        str = "0.00";
    }
    return str;
}

std::string format_decimal(int val) {
    std::ostringstream o;
    o << val;
    std::string s(o.str());
    std::string str;
    str.reserve(s.size() + 8);
    for (size_t i = 0; i < s.size(); i++) {
        if ((s.size() - i) % 3 == 0 && i != 0 && s[i - 1] != '-') {
            str += ',';
        }
        str += s[i];
    }
    return str;
}

std::vector<std::string> strsplit(const std::string& str, const char sep /* =? */) {
    std::vector<std::string> split;
    size_t begin, end;
    begin = str.find_first_not_of(sep);
    while ((end = str.find(sep, begin)) != std::string::npos) {
        split.push_back(str.substr(begin, end - begin));
        begin = str.find_first_not_of(sep, end);
    }
    if (begin != std::string::npos && begin < str.size()) {
        split.push_back(str.substr(begin));
    }
    return split;
}

} // namespace base
