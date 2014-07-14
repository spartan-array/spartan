#ifndef __CONFIG_H__
#define __CONFIG_H__
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <map>

#include <cassert>


class Flag {
public:
    std::string class_name;
    std::string name;
    std::string help;
    std::string val_str;

    Flag() : name(""), help("") {};
    Flag(std::string name, std::string default_val = "", std::string help = "") {
        this->name = name;
        this->help = help;
        this->val_str = default_val;
    };

    std::string get_val_str(void) {return val_str;};
    virtual void parse(std::string str) {return;};
};

class IntFlag : public Flag {
public:
    int val;
    IntFlag(std::string name, std::string default_val = "", std::string help = "")
        : Flag(name, default_val, help) {
        parse(default_val);
        class_name = "IntFlag";
    };
    void parse(std::string str) {
        std::istringstream(str) >> val;
    };

    int get(void) {return val;};
};

class StrFlag : public Flag {
public:
    std::string val;
    StrFlag(std::string name, std::string default_val = "", std::string help = "")
        : Flag(name, default_val, help) {
        parse(default_val);
        class_name = "StrFlag";
    };
    void parse(std::string str) {
        val = str;
    }
    std::string get(void) {return val;};
};

class BoolFlag : public Flag {
public:
    bool val;
    BoolFlag(std::string name, std::string default_val = "", std::string help = "")
        : Flag(name, default_val, help) {
        parse(default_val);
        class_name = "BoolFlag";
    };
    void parse(std::string str) {
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
        if (str == "false" || str == "0") {
            val = false;
        } else if(str == "true" || str == "1") {
            val = true;
        } else {
           assert (0);
        }
    };
    bool get(void) {return val;};
};

struct host {
    std::string name;
    int count;
};
class HostListFlag : public Flag {
public:
    std::vector <struct host> val;
    HostListFlag(std::string name, std::string default_val = "", std::string help = "")
        : Flag(name, default_val, help) {
        parse(default_val);
        class_name = "HostListFlag";
    };
    void parse(std::string str) {
        std::istringstream ss(str);
        std::string host;

        while (std::getline(ss, host, ',')) {
            std::string hostname;
            int hostcount;
            size_t split = host.find(":");

            hostname = host.substr(0, split);
            std::istringstream (host.substr(split + 1)) >> hostcount;

            struct host h = {.name = hostname, .count = hostcount}              ;
            val.push_back(h);
        }
    };
    struct host get(int index) {return val[index];};
};

enum AssignMode {
    BY_CORE = 1,
    BY_NODE = 2,
};

enum LogLevel {
    LOGLEVEL_DEBUG = 1,
    LOGLEVEL_INFO = 2,
    LOGLEVEL_WARN = 3,
    LOGLEVEL_ERROR = 4,
    LOGLEVEL_FATAL = 5,
};

class AssignModeFlag : public Flag {
public:
    AssignMode val;

    AssignModeFlag(std::string name, std::string default_val = "", std::string help = "")
        : Flag(name, default_val, help) {
        parse(default_val);
        class_name = "AssignModeFlag";
    };
    void parse(std::string str) {
        if (str == "BY_CORE") {
            val = BY_CORE;
        } else if (str == "BY_NODE") {
            val = BY_NODE; 
        } else {
            assert(0);
        }
    };
    AssignMode get(void) {return val;};
};

class LogLevelFlag : public Flag {
public:
    LogLevel val;
    LogLevelFlag(std::string name, std::string default_val = "", std::string help = "")
        : Flag(name, default_val, help) {
        parse(default_val);
        class_name = "LogLevelFlag";
    };
    void parse(std::string str) {
        if (str == "DEBUG") {
            val = LOGLEVEL_DEBUG;
        } else if(str == "INFO") {
            val = LOGLEVEL_INFO;
        } else if(str == "WARN") {
            val = LOGLEVEL_WARN;
        } else if(str == "ERROR") {
            val = LOGLEVEL_ERROR;
        } else if(str == "FATAL") {
            val = LOGLEVEL_FATAL;
        } else {
            assert(0);
        }
    };
    LogLevel get(void) {return val;};
};

class Flags {
private:
    bool parsed;
    std::map<std::string, Flag*> vals;
    std::map<std::string, Flag*>::iterator next_it;

public:
    Flags(void) {
        parsed = false;
        next_it = vals.begin();
    };

    ~Flags() {};

    void add(Flag *flag) {
        vals[flag->name] = flag;
    };

    Flag* get(std::string keyword) {
        if (vals.find(keyword) != vals.end()) {
            return vals[keyword];
        } else {
            return NULL;
        }
    };

    Flag* next(void) {
        if (next_it == vals.end()) {
            return NULL;
        } else {
            Flag *ret = next_it->second;
            next_it++;
            return ret;
        }
    };

    void reset_next(void) {
        next_it = vals.begin();
    };

    bool is_parsed(void) {
        return parsed;
    };

    int get_flag_count(void) {
        return vals.size();
    };

    void set_parsed(void) {
        parsed = true;
    };

};

void init_flags(void);
extern Flags FLAGS;
#endif
