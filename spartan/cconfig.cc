/**
 * C++ implementation of configuration options and flags.
 */


#include <string>
#include <sstream>
#include <map>
#include <vector>
#include "cconfig.h"

CFlags FLAGS; 
void init_flags(void) 
{
    static bool done = false;
    if (done) 
        return;

    FLAGS.add(new BoolFlag("print_options", "false"));
    FLAGS.add(new BoolFlag("profile_worker", "false"));
    FLAGS.add(new BoolFlag("profile_master", "false"));
    FLAGS.add(new BoolFlag("cluster", "false"));
    FLAGS.add(new LogLevelFlag("log_level", "INFO"));
    FLAGS.add(new IntFlag("num_workers", "3"));
    FLAGS.add(new IntFlag("port_base"," 10000", "Port master should listen on"));
    FLAGS.add(new StrFlag("tile_assignment_strategy", "round_robin",
                      "Decide tile to worker mapping (round_robin, random, performance)"));
    FLAGS.add(new StrFlag("checkpoint_path", "/tmp/spartan/checkpoint/", 
                      "Path for saving checkpoint information"));
    FLAGS.add(new IntFlag("default_rpc_timeout", "60"));
    FLAGS.add(new IntFlag("max_zeromq_sockets", "4096"));

    FLAGS.add(new BoolFlag("opt_keep_stack", "false"));
    FLAGS.add(new BoolFlag("capture_expr_stack", "false"));
    FLAGS.add(new BoolFlag("dump_timers", "false"));
    FLAGS.add(new BoolFlag("load_balance", "false"));

    FLAGS.add(new HostListFlag("hosts", "localhost:8"));
    FLAGS.add(new BoolFlag("xterm", "false", "Run workers in xterm"));
    FLAGS.add(new BoolFlag("oprofile", "false", "Run workers inside of operf"));
    FLAGS.add(new AssignModeFlag("assign_mode", "BY_NODE"));
    FLAGS.add(new BoolFlag("use_single_core", "true"));

    FLAGS.add(new BoolFlag("use_threads", "true",
                       "When running locally, use threads instead of forking."
                          "(slow, for debugging)"));
    FLAGS.add(new IntFlag("heartbeat_interval", "3", "Heartbeat Interval in each worker"));
    FLAGS.add(new IntFlag("worker_failed_heartbeat_threshold", "10", 
                      "the max number of heartbeat that a worker can delay"));
    FLAGS.add(new BoolFlag("optimization", "true"));

    FLAGS.add(new BoolFlag("opt_expression_cache", "true", "Enable expression caching."));

    FLAGS.add(new BoolFlag("dump_timers", "false"));

    FLAGS.add(new BoolFlag("use_cuda", "false"));
}

std::map<std::string, std::string> parse_argv(int argc, const char **argv)
{
    std::map<std::string, std::string> argv_map; 
    std::string flag;
    std::string name, val;
    size_t split;
    int i;
    
    for (i = 0; i < argc; i++) {
        if (flag[0] != '-') {
           continue; 
        }
        flag = std::string(argv[i]);
        split = flag.find('=');
        name = flag.substr(2, split - 2);
        val = flag.substr(split + 1);
        if (val[0] == '\'' || val[0] == '\"') {
            val = val.substr(1, val.size() - 2);
        }
        argv_map[name] = val;
    }

    return argv_map;
}

/**
 * There are some assumptions for this function:
 *   1. This is only called from workers (the master should call python version).
 *   2. Workers needn't report parse errors to users (the master should do this).
 *   3. This API is only called in the beginning of execution. 
 *      So it can use Python libraries without interfering the 
 *      python computation thread.
 *   4. The master can read the config file while workers can only read from
 *      arguments. And the master add configurations in the config file to 
 *      arguments. (See cluster.py)
 */
void config_parse(int argc, const char **argv)
{
    if (FLAGS.is_parsed()) {
        return;
    }

    init_flags();
    FLAGS.set_parsed();

    CFlag* flag;
    std::map<std::string, std::string> argv_map;
    std::string val;
   
    argv_map = parse_argv(argc, &argv[0]);
    for (std::map<std::string, std::string>::iterator it = argv_map.begin(); it != argv_map.end(); ++it) {
        if ((flag = FLAGS.get(it->first)) != NULL) {
            flag->parse(it->second);
        }
    }
}

//void get_flags_info(std::vector<const char*>* list)
std::vector<const char*> get_flags_info(void)
{
    std::vector<const char*> list;
    CFlag* flag;

    init_flags();
    FLAGS.reset_next();
    while ((flag = FLAGS.next()) != NULL) {
        list.push_back(flag->class_name.c_str());
        list.push_back(flag->name.c_str());
        list.push_back(flag->val_str.c_str());
        list.push_back(flag->help.c_str());
    }
    return list;
}

#ifdef __UNIT_TEST__
#include <iostream>
int main(int argc, const char *argv[])
{
    config_parse(argc, argv);
    return 0;
}
#endif
