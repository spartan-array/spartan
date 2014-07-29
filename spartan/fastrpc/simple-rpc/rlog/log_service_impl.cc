#include <string>
#include <sstream>

#include <time.h>
#include <sys/time.h>

#include "log_service_impl.h"

using namespace rpc;
using namespace std;
using namespace rlog;

namespace rlog {

RLogServiceImpl::RLogServiceImpl() {
    qps_interval_ = 1.0;
    char* s = getenv("QPS_INTERVAL");
    if (s != nullptr) {
        qps_interval_ = strtod(s, nullptr);
        Log::info("qps interval set to %lf", qps_interval_);
    }
}

void RLogServiceImpl::log(const i32& level, const std::string& source, const i64& msg_id, const std::string& message) {
    log_piece piece;
    piece.msg_id = msg_id;
    piece.level = level;
    piece.message = message;

    char tm_str[TIME_NOW_STR_SIZE];
    base::time_now_str(tm_str);

    l_.lock();

    set<log_piece>& buffer = buffer_[source];
    buffer.insert(piece);

    i64& done = done_[source];

    while (!buffer.empty() && buffer.begin()->msg_id <= done + 1) {
        LOG_INFO("level=%d %s %s: %s", buffer.begin()->level, tm_str, source.c_str(), buffer.begin()->message.c_str());
        done = max(done, buffer.begin()->msg_id);
        buffer.erase(buffer.begin());
    }

    l_.unlock();
}

void RLogServiceImpl::aggregate_qps(const std::string& metric_name, const rpc::i32& incr) {
    agg_qps_l_.lock();
    list<agg_qps_record>& records = agg_qps_[metric_name];
    agg_qps_record new_rec;
    timeval tv;
    gettimeofday(&tv, nullptr);
    double now = tv.tv_sec + tv.tv_usec / 1000.0 / 1000.0;
    new_rec.tm = now;
    if (records.empty()) {
        new_rec.agg_count = incr;
    } else {
        new_rec.agg_count = records.back().agg_count + incr;
    }
    records.push_back(new_rec);

    // only keep 5min qps info
    while (now - records.front().tm > 5 * 60) {
        records.erase(records.begin());
    }

    // report aggregate qps every 1sec
    if (now - last_qps_tm_[metric_name] >= qps_interval_) {
        const int report_intervals[] = {1, 5, 15, 30, 60};
        ostringstream qps_ostr;
        for (size_t i = 0; i < arraysize(report_intervals); i++) {
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
        last_qps_tm_[metric_name] = now;
        if (qps_ostr.str().length() > 0) {
            Log_info("qps '%s':%s", metric_name.c_str(), qps_ostr.str().c_str());
        }
    }

    agg_qps_l_.unlock();
}

} // namespace rlog
