#include <string>
#include <sstream>

#include <time.h>
#include <sys/time.h>

#include "log_service_impl.h"

using namespace rpc;
using namespace std;
using namespace logservice;

void RLogServiceImpl::log(const i32& level, const std::string& source, const i64& msg_id, const std::string& message) {
    log_piece piece;
    piece.msg_id = msg_id;
    piece.level = level;
    piece.message = message;

    const int tm_str_len = 80;
    char tm_str[tm_str_len];
    time_t now = time(NULL);
    struct tm tm_val;
    localtime_r(&now, &tm_val);
    strftime(tm_str, tm_str_len - 1, "%F %T", &tm_val);
    timeval tv;
    gettimeofday(&tv, NULL);

    l_.lock();

    set<log_piece>& buffer = buffer_[source];
    buffer.insert(piece);

    i64& done = done_[source];

    while (!buffer.empty() && buffer.begin()->msg_id <= done + 1) {
        Log::log(buffer.begin()->level, "%s.%03d %s: %s", tm_str, tv.tv_usec / 1000, source.c_str(), buffer.begin()->message.c_str());
        done = max(done, buffer.begin()->msg_id);
        buffer.erase(buffer.begin());
    }

    l_.unlock();
}

void RLogServiceImpl::aggregate_qps(const std::string& metric_name, const rpc::i32& increment) {
    agg_qps_l_.lock();
    list<agg_qps_record>& records = agg_qps_[metric_name];
    agg_qps_record new_rec;
    timeval tv;
    gettimeofday(&tv, NULL);
    double now = tv.tv_sec + tv.tv_usec / 1000.0 / 1000.0;
    new_rec.tm = now;
    if (records.empty()) {
        new_rec.agg_count = increment;
    } else {
        new_rec.agg_count = records.back().agg_count + increment;
    }
    records.push_back(new_rec);

    // only keep 5min qps info
    while (now - records.front().tm > 5 * 60) {
        records.erase(records.begin());
    }

    // report aggregate qps every 1sec
    if (now - last_qps_report_tm_ > 1) {
        const int report_intervals[] = {1, 5, 15, 30, 60};
        ostringstream qps_ostr;
        for (size_t i = 0; i < sizeof(report_intervals) / sizeof(report_intervals[0]); i++) {
            double increment = -1;
            const int report_interval = report_intervals[i];
            for (list<agg_qps_record>::reverse_iterator it = records.rbegin(); it != records.rend(); ++it) {
                if (now - it->tm < report_interval) {
                    increment = records.back().agg_count - it->agg_count;
                } else {
                    break;
                }
            }
            if (increment > 0) {
                double qps = increment / report_interval;
                qps_ostr << " " << report_interval << ":" << qps;
            }
        }
        last_qps_report_tm_ = now;
        if (qps_ostr.str().length() > 0) {
            Log::info("qps '%s':%s", metric_name.c_str(), qps_ostr.str().c_str());
        }
    }

    agg_qps_l_.unlock();
}
