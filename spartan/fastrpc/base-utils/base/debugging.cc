#include <execinfo.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <utility>

#include "debugging.h"
#include "misc.h"

using namespace std;

namespace base {

#ifdef __APPLE__

void print_stack_trace(FILE* fp /* =? */) {
    const int max_trace = 1024;
    void* callstack[max_trace];
    memset(callstack, 0, sizeof(callstack));
    int frames = backtrace(callstack, max_trace);

    char **str_frames = backtrace_symbols(callstack, frames);
    if (str_frames == nullptr) {
        fprintf(fp, "  *** failed to obtain stack trace!\n");
        return;
    }

    fprintf(fp, "  *** begin stack trace ***\n");
    for (int i = 0; i < frames - 1; i++) {
        string trace = str_frames[i];
        size_t idx = trace.rfind(' ');
        size_t idx2 = trace.rfind(' ', idx - 1);
        idx = trace.rfind(' ', idx2 - 1) + 1;
        string mangled = trace.substr(idx, idx2 - idx);
        string left = trace.substr(0, idx);
        string right = trace.substr(idx2);

        string cmd = "c++filt -n ";
        cmd += mangled;

        auto demangle = popen(cmd.c_str(), "r");
        if (demangle) {
            string demangled = getline(demangle);
            fprintf(fp, "%s%s%s\n", left.c_str(), demangled.c_str(), right.c_str());
            pclose(demangle);
        } else {
            fprintf(fp, "%s\n", str_frames[i]);
        }
    }
    fprintf(fp, "  ***  end stack trace  ***\n");
    fflush(fp);

    free(str_frames);
}

#else // no __APPLE__

void print_stack_trace(FILE* fp /* =? */) {
    const int max_trace = 1024;
    void* callstack[max_trace];
    memset(callstack, 0, sizeof(callstack));
    int frames = backtrace(callstack, max_trace);

    char **str_frames = backtrace_symbols(callstack, frames);
    if (str_frames == nullptr) {
        fprintf(fp, "  *** failed to obtain stack trace!\n");
        return;
    }

    fprintf(fp, "  *** begin stack trace ***\n");
    const char* exec_path = get_exec_path();
    vector<pair<string, string>> fmt_output;
    size_t max_func_length = 0;
    for (int i = 0; i < frames - 1; i++) {
        bool addr2line_ok = false;
        if (exec_path != nullptr) {
            char buf[32];
            snprintf(buf, sizeof(buf), "addr2line %p -e ", callstack[i]);
            string cmd = buf;
            cmd += exec_path;
            cmd += " -f -C 2>&1";
            auto addr2line = popen(cmd.c_str(), "r");
            if (addr2line) {
                addr2line_ok = true;
                string demangled_func_name = getline(addr2line);
                if (demangled_func_name[0] == '?') {
                    addr2line_ok = false;
                } else {
                    max_func_length = max(max_func_length, demangled_func_name.size());
                    string file_line = getline(addr2line);
                    fmt_output.push_back(make_pair(demangled_func_name, file_line));
                }
                pclose(addr2line);
            }
        }
        if (!addr2line_ok) {
            max_func_length = max(max_func_length, strlen(str_frames[i]));
            fmt_output.push_back(make_pair(str_frames[i], ""));
        }
    }
    for (size_t i = 0; i < fmt_output.size(); i++) {
        fprintf(fp, "%-3lu  %s", i, fmt_output[i].first.c_str());
        if (fmt_output[i].second.size() > 0) {
            int padding = max_func_length - fmt_output[i].first.size() + 4;
            while (padding > 0) {
                padding--;
                fputc(' ', fp);
            }
            fprintf(fp, "%s\n", fmt_output[i].second.c_str());
        } else {
            fputc('\n', fp);
        }
    }

    fprintf(fp, "  ***  end stack trace  ***\n");
    fflush(fp);

    free(str_frames);
}

#endif // ifdef __APPLE__

} // namespace base
