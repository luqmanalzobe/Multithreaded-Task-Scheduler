// src/main.cpp
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using ms    = std::chrono::milliseconds;

struct Task {
    std::string id;
    int arrival_ms{};
    int burst_ms{};
    int priority{};          // lower value = higher priority (1 > 2)
    int remaining_ms{};
    int start_time_ms{-1};   // first time on CPU
    int finish_time_ms{-1};  // completion time
    int total_wait_ms{0};    // aggregated waiting
};

enum class Algo { RR, PRIORITY };

struct Args {
    Algo algo = Algo::RR;
    int quantum_ms = 50;
    std::string tasks_path; // optional
};

static std::string pad(int t) {
    std::ostringstream oss;
    oss << std::setw(3) << std::setfill('0') << t;
    return oss.str();
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--algo" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "rr") a.algo = Algo::RR;
            else if (v == "priority") a.algo = Algo::PRIORITY;
            else std::cerr << "Unknown algo, defaulting to rr\n";
        } else if (s == "--quantum" && i + 1 < argc) {
            a.quantum_ms = std::max(1, std::stoi(argv[++i]));
        } else if (s == "--tasks" && i + 1 < argc) {
            a.tasks_path = argv[++i];
        } else {
            std::cerr << "Unknown arg: " << s << "\n";
        }
    }
    return a;
}

static std::vector<Task> default_tasks() {
    return {
        {"T1", 0,   120, 2, 120},
        {"T2", 10,  200, 1, 200},
        {"T3", 80,  150, 3, 150},
        {"T4", 120, 90,  2,  90},
    };
}

static std::vector<Task> load_tasks_csv(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "Could not open tasks CSV: " << path << " — using defaults.\n";
        return default_tasks();
    }
    std::vector<Task> tasks;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        // skip header if present
        if (line.find("id") != std::string::npos) continue;
        std::istringstream ss(line);
        std::string id; int arr, burst, prio;
        char comma;
        if (std::getline(ss, id, ',') &&
            ss >> arr >> comma >> burst >> comma >> prio) {
            Task t{id, arr, burst, prio, burst};
            tasks.push_back(t);
        }
    }
    if (tasks.empty()) return default_tasks();
    return tasks;
}

struct Log {
    std::mutex m;
    std::vector<std::string> lines;
    void add(const std::string& s) {
        std::lock_guard<std::mutex> lk(m);
        lines.push_back(s);
    }
    void flush() {
        for (auto& l : lines) std::cout << l << "\n";
    }
};

struct ReadyItem {
    int priority; // for priority queue (lower is better)
    int arrival_order;
    Task* task;
};

struct ReadyComparePriority {
    bool operator()(const ReadyItem& a, const ReadyItem& b) const {
        if (a.priority != b.priority) return a.priority > b.priority; // min-heap behavior
        return a.arrival_order > b.arrival_order;
    }
};

class Scheduler {
  public:
    Scheduler(Args args, std::vector<Task> tasks)
        : args_(args), all_tasks_(std::move(tasks)) {
        std::sort(all_tasks_.begin(), all_tasks_.end(),
                  [](const Task& a, const Task& b){return a.arrival_ms < b.arrival_ms;});
    }

    void run() {
        start_ = Clock::now();
        std::thread prod(&Scheduler::producer, this);
        std::thread cpu(&Scheduler::cpu_worker, this);

        prod.join();
        {
            std::lock_guard<std::mutex> lk(mx_);
            producer_done_ = true;
        }
        cv_.notify_all();
        cpu.join();

        print_metrics();
    }

  private:
    Args args_;
    std::vector<Task> all_tasks_;
    std::map<std::string, Task*> task_index_;
    std::atomic<bool> producer_done_{false};

    // timing
    Clock::time_point start_;
    int now_ms() const {
        return (int)std::chrono::duration_cast<ms>(Clock::now() - start_).count();
    }
    void sleep_ms(int t) { std::this_thread::sleep_for(ms(t)); }

    // ready queues
    std::mutex mx_;
    std::condition_variable cv_;
    int arrival_seq_ = 0;

    std::queue<Task*> rr_queue_;
    std::priority_queue<ReadyItem, std::vector<ReadyItem>, ReadyComparePriority> prio_queue_;

    Log log_;

    void producer() {
        // create quick index
        for (auto& t : all_tasks_) task_index_[t.id] = &t;

        // simulate arrival by sleeping until arrival_ms
        int prev = 0;
        for (auto& t : all_tasks_) {
            int wait = std::max(0, t.arrival_ms - prev);
            sleep_ms(wait);
            prev = t.arrival_ms;

            {
                std::lock_guard<std::mutex> lk(mx_);
                t.remaining_ms = t.burst_ms;
                if (args_.algo == Algo::RR) {
                    rr_queue_.push(&t);
                } else {
                    prio_queue_.push({t.priority, arrival_seq_++, &t});
                }
                log_.add("[" + pad(now_ms()) + "ms] Arrived: " + t.id +
                         " (burst=" + std::to_string(t.burst_ms) +
                         ", prio=" + std::to_string(t.priority) + ")");
            }
            cv_.notify_all();
        }
    }

    void cpu_worker() {
        if (args_.algo == Algo::RR) run_rr();
        else run_priority();
    }

    void run_rr() {
        const int q = args_.quantum_ms;
        int last_check = 0;
        // track last seen time per task to compute waiting increments
        std::map<std::string, int> last_ready_time;

        while (true) {
            Task* t = nullptr;
            {
                std::unique_lock<std::mutex> lk(mx_);
                cv_.wait(lk, [&]{ return !rr_queue_.empty() || producer_done_; });

                if (rr_queue_.empty() && producer_done_) break;

                t = rr_queue_.front();
                rr_queue_.pop();

                // waiting time since it was enqueued
                int now = now_ms();
                if (last_ready_time.count(t->id)) {
                    t->total_wait_ms += now - last_ready_time[t->id];
                } else {
                    // first time getting CPU — waiting = now - arrival
                    t->total_wait_ms += now - t->arrival_ms;
                }
            }

            if (t->start_time_ms < 0) t->start_time_ms = now_ms();

            int slice = std::min(q, t->remaining_ms);
            log_.add("[" + pad(now_ms()) + "ms] CPU -> " + t->id +
                     " (slice=" + std::to_string(slice) + ") rem=" +
                     std::to_string(t->remaining_ms - slice));
            sleep_ms(slice);
            t->remaining_ms -= slice;

            {
                std::lock_guard<std::mutex> lk(mx_);
                if (t->remaining_ms > 0) {
                    // requeue and mark time it became ready again
                    rr_queue_.push(t);
                    last_ready_time[t->id] = now_ms();
                } else {
                    t->finish_time_ms = now_ms();
                    log_.add("[" + pad(t->finish_time_ms) + "ms] Done: " + t->id +
                             " (wait=" + std::to_string(t->total_wait_ms) +
                             ", turnaround=" + std::to_string(t->finish_time_ms - t->arrival_ms) + ")");
                }
            }
            cv_.notify_all();
            last_check = now_ms();
        }
    }

    void run_priority() {
        // Non-preemptive priority: always run the highest priority (lowest number) to completion.
        // Waiting time accumulated while sitting in the ready queue.
        std::map<std::string, int> last_ready_time;

        while (true) {
            Task* t = nullptr;
            {
                std::unique_lock<std::mutex> lk(mx_);
                cv_.wait(lk, [&]{ return !prio_queue_.empty() || producer_done_; });
                if (prio_queue_.empty() && producer_done_) break;

                auto top = prio_queue_.top(); prio_queue_.pop();
                t = top.task;

                int now = now_ms();
                if (last_ready_time.count(t->id)) {
                    t->total_wait_ms += now - last_ready_time[t->id];
                } else {
                    t->total_wait_ms += now - t->arrival_ms;
                }
            }

            if (t->start_time_ms < 0) t->start_time_ms = now_ms();

            log_.add("[" + pad(now_ms()) + "ms] CPU -> " + t->id +
                     " (run-to-complete) rem=" + std::to_string(t->remaining_ms - t->remaining_ms));
            // run to completion (simulate work)
            sleep_ms(t->remaining_ms);
            t->remaining_ms = 0;
            t->finish_time_ms = now_ms();

            log_.add("[" + pad(t->finish_time_ms) + "ms] Done: " + t->id +
                     " (wait=" + std::to_string(t->total_wait_ms) +
                     ", turnaround=" + std::to_string(t->finish_time_ms - t->arrival_ms) + ")");
            cv_.notify_all();
        }
    }

    void print_metrics() {
        log_.flush();
        // compute averages
        int n = 0;
        double sum_wait = 0, sum_turn = 0;
        for (auto& kv : task_index_) {
            Task* t = kv.second;
            if (t->finish_time_ms < 0) continue;
            n++;
            sum_wait += t->total_wait_ms;
            sum_turn += (t->finish_time_ms - t->arrival_ms);
        }
        if (n == 0) return;
        std::cout << "---- Metrics ----\n";
        std::cout << "Average waiting time   : " << (sum_wait / n) << " ms\n";
        std::cout << "Average turnaround time: " << (sum_turn / n) << " ms\n";
    }
};

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    auto tasks = args.tasks_path.empty() ? default_tasks() : load_tasks_csv(args.tasks_path);
    Scheduler sched(args, std::move(tasks));
    sched.run();
    return 0;
}
