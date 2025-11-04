// -------------------------------------------------------------
// Multithreaded CPU Scheduler Simulation
// - Simulates Round-Robin (preemptive, time-sliced) and
//   non-preemptive Priority scheduling.
// - Two threads:
//     * Producer: releases tasks at their arrival times
//     * CPU worker: executes tasks according to selected policy
// - Collects per-task metrics and prints averages at the end.
// -------------------------------------------------------------

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

using Clock = std::chrono::steady_clock;    // monotonic clock for timing
using ms    = std::chrono::milliseconds;    // convenience alias

// -----------------------------
// Task model (one job in system)
// -----------------------------
struct Task {
    std::string id;        // human-readable identifier (e.g., "T1")
    int arrival_ms{};      // when this task enters the system (ms from sim start)
    int burst_ms{};        // total CPU time needed to complete (ms)
    int priority{};        // lower number = higher priority (1 beats 2)
    int remaining_ms{};    // remaining CPU time (counts down to 0)
    int start_time_ms{-1}; // first time the task ever got CPU (ms from sim start)
    int finish_time_ms{-1};// when task completed (ms from sim start)
    int total_wait_ms{0};  // aggregated time spent waiting in the ready queue
};

// -----------------------------
// Algorithm selector (scoped enum)
// -----------------------------
enum class Algo { RR, PRIORITY };

// --------------------------------------
// Run-time configuration (from CLI args)
// --------------------------------------
struct Args {
    Algo algo = Algo::RR;      // default to Round Robin
    int quantum_ms = 50;       // RR time slice (ms)
    std::string tasks_path;    // optional CSV path; empty => defaults
};

// Format helper: zero-pad small numbers to width=3 (e.g., 5 -> "005")
static std::string pad(int t) {
    std::ostringstream oss;
    oss << std::setw(3) << std::setfill('0') << t;
    return oss.str();
}

// -----------------------------------------------------
// Parse command-line args into an Args configuration
// Supported flags:
//   --algo rr|priority
//   --quantum <ms>
//   --tasks <path>
// -----------------------------------------------------
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

// ---------------------------------------
// Built-in demo workload (when no CSV)
// ---------------------------------------
static std::vector<Task> default_tasks() {
    return {
        {"T1", 0,   120, 2, 120},
        {"T2", 10,  200, 1, 200},
        {"T3", 80,  150, 3, 150},
        {"T4", 120, 90,  2,  90},
    };
}

// -----------------------------------------------------
// Load tasks from a CSV: id,arrival_ms,burst_ms,priority
// Falls back to defaults if file can't be parsed.
// -----------------------------------------------------
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
        // Skip header line if present (contains "id")
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

// -------------------------------------------------------
// Thread-safe log collector:
// - add(): append a line under a mutex
// - flush(): print collected lines (called after threads)
// -------------------------------------------------------
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

// -----------------------------------------------------------------
// Data stored in the Priority ready-queue (min-heap by comparator):
//   - priority: lower number wins (more important)
//   - arrival_order: FIFO tie-break among same-priority tasks
//   - task: pointer to canonical Task (no copies of Task objects)
// -----------------------------------------------------------------
struct ReadyItem {
    int priority;      // for priority queue (lower is better)
    int arrival_order; // FIFO among equals
    Task* task;
};

// Comparator to invert std::priority_queue into a min-heap by priority
struct ReadyComparePriority {
    bool operator()(const ReadyItem& a, const ReadyItem& b) const {
        if (a.priority != b.priority) return a.priority > b.priority; // lower first
        return a.arrival_order > b.arrival_order;                      // earlier first
    }
};

// =====================================================
// Scheduler: orchestrates producer & CPU worker threads
// =====================================================
class Scheduler {
  public:
    // Constructor: store args/tasks, then sort tasks by arrival time
    Scheduler(Args args, std::vector<Task> tasks)
        : args_(args), all_tasks_(std::move(tasks)) {
        std::sort(all_tasks_.begin(), all_tasks_.end(),
                  [](const Task& a, const Task& b){ return a.arrival_ms < b.arrival_ms; });
    }

    // Entry point: launch threads, wait, then report metrics
    void run() {
        start_ = Clock::now();
        std::thread prod(&Scheduler::producer, this);     // releases tasks over time
        std::thread cpu (&Scheduler::cpu_worker, this);   // executes tasks

        // Wait for producer to finish releasing all tasks
        prod.join();
        {
            std::lock_guard<std::mutex> lk(mx_);
            producer_done_ = true;                        // signal no more arrivals
        }
        cv_.notify_all();                                 // wake worker if it's waiting
        cpu.join();                                       // wait for all work to finish

        print_metrics();                                  // print timeline + averages
    }

  private:
    // -----------------
    // Internal state
    // -----------------
    Args args_;
    std::vector<Task> all_tasks_;               // canonical storage of tasks
    std::map<std::string, Task*> task_index_;   // id -> Task* for reporting
    std::atomic<bool> producer_done_{false};    // set when producer exits

    // Timing helpers
    Clock::time_point start_;
    int now_ms() const {
        return (int)std::chrono::duration_cast<ms>(Clock::now() - start_).count();
    }
    void sleep_ms(int t) { std::this_thread::sleep_for(ms(t)); }

    // Ready queues + sync primitives
    std::mutex mx_;
    std::condition_variable cv_;
    int arrival_seq_ = 0;  // increases per enqueue (FIFO tie-break in priority)

    std::queue<Task*> rr_queue_;   // FIFO for Round-Robin
    std::priority_queue<ReadyItem, std::vector<ReadyItem>, ReadyComparePriority> prio_queue_;

    Log log_;  // thread-safe timeline

    // ----------------------------------------------
    // Producer thread: release tasks by arrival time
    // ----------------------------------------------
    void producer() {
        // Build index for metrics reporting
        for (auto& t : all_tasks_) task_index_[t.id] = &t;

        // Sleep deltas between sorted arrivals, then enqueue
        int prev = 0;
        for (auto& t : all_tasks_) {
            int wait = std::max(0, t.arrival_ms - prev);
            sleep_ms(wait);
            prev = t.arrival_ms;

            {
                std::lock_guard<std::mutex> lk(mx_);
                t.remaining_ms = t.burst_ms; // reset on entry
                if (args_.algo == Algo::RR) {
                    rr_queue_.push(&t);
                } else {
                    prio_queue_.push({t.priority, arrival_seq_++, &t});
                }
                log_.add("[" + pad(now_ms()) + "ms] Arrived: " + t.id +
                         " (burst=" + std::to_string(t.burst_ms) +
                         ", prio=" + std::to_string(t.priority) + ")");
            }
            cv_.notify_all(); // wake worker to consider new arrivals
        }
    }

    // --------------------------------------------
    // CPU worker: route to selected scheduling algo
    // --------------------------------------------
    void cpu_worker() {
        if (args_.algo == Algo::RR) run_rr();
        else                         run_priority();
    }

    // ------------------------------------------------------
    // Round-Robin (preemptive): FIFO queue + fixed quantum
    // ------------------------------------------------------
    void run_rr() {
        const int q = args_.quantum_ms;

        // Track when a task (by id) last became ready to compute waiting increments
        std::map<std::string, int> last_ready_time;

        while (true) {
            Task* t = nullptr;
            {
                std::unique_lock<std::mutex> lk(mx_);
                // Wait until there is work OR the producer is done
                cv_.wait(lk, [&]{ return !rr_queue_.empty() || producer_done_; });

                // If queue is empty and no more arrivals will come, we're done
                if (rr_queue_.empty() && producer_done_) break;

                // Pop the next ready task (FIFO)
                t = rr_queue_.front();
                rr_queue_.pop();

                // Compute waiting time increment
                int now = now_ms();
                if (last_ready_time.count(t->id)) {
                    t->total_wait_ms += now - last_ready_time[t->id];
                } else {
                    // First CPU grant: waited from arrival to now
                    t->total_wait_ms += now - t->arrival_ms;
                }
            }

            // First time this task ever got CPU?
            if (t->start_time_ms < 0) t->start_time_ms = now_ms();

            // Execute one time slice
            int slice = std::min(q, t->remaining_ms);
            log_.add("[" + pad(now_ms()) + "ms] CPU -> " + t->id +
                     " (slice=" + std::to_string(slice) + ") rem=" +
                     std::to_string(t->remaining_ms - slice));

            // Simulate running without holding the lock
            sleep_ms(slice);
            t->remaining_ms -= slice;

            {
                std::lock_guard<std::mutex> lk(mx_);
                if (t->remaining_ms > 0) {
                    // Not finished: reinsert at tail and mark it "ready again" now
                    rr_queue_.push(t);
                    last_ready_time[t->id] = now_ms();
                } else {
                    // Finished: record finish time and log stats
                    t->finish_time_ms = now_ms();
                    log_.add("[" + pad(t->finish_time_ms) + "ms] Done: " + t->id +
                             " (wait=" + std::to_string(t->total_wait_ms) +
                             ", turnaround=" + std::to_string(t->finish_time_ms - t->arrival_ms) + ")");
                }
            }
            cv_.notify_all(); // wake any waiting worker in edge cases
        }
    }

    // -----------------------------------------------------------------
    // Non-preemptive Priority: always run lowest priority-number first
    // -----------------------------------------------------------------
    void run_priority() {
        // Track when a task last became ready (for correct waiting sums)
        std::map<std::string, int> last_ready_time;

        while (true) {
            Task* t = nullptr;
            {
                std::unique_lock<std::mutex> lk(mx_);
                // Wait until there is work OR the producer is done
                cv_.wait(lk, [&]{ return !prio_queue_.empty() || producer_done_; });
                if (prio_queue_.empty() && producer_done_) break;

                // Pick highest-priority task (lowest numeric value), FIFO on ties
                auto top = prio_queue_.top(); prio_queue_.pop();
                t = top.task;

                // Compute waiting time increment
                int now = now_ms();
                if (last_ready_time.count(t->id)) {
                    t->total_wait_ms += now - last_ready_time[t->id];
                } else {
                    t->total_wait_ms += now - t->arrival_ms;
                }
            }

            // First CPU grant?
            if (t->start_time_ms < 0) t->start_time_ms = now_ms();

            // Log remaining BEFORE running (clarity fix)
            log_.add("[" + pad(now_ms()) + "ms] CPU -> " + t->id +
                     " (run-to-complete) rem=" + std::to_string(t->remaining_ms));

            // Run to completion (no preemption)
            sleep_ms(t->remaining_ms);
            t->remaining_ms = 0;
            t->finish_time_ms = now_ms();

            log_.add("[" + pad(t->finish_time_ms) + "ms] Done: " + t->id +
                     " (wait=" + std::to_string(t->total_wait_ms) +
                     ", turnaround=" + std::to_string(t->finish_time_ms - t->arrival_ms) + ")");
            cv_.notify_all();
        }
    }

    // ------------------------------------------
    // Print timeline and average performance stats
    // ------------------------------------------
    void print_metrics() {
        log_.flush();

        int n = 0;
        double sum_wait = 0, sum_turn = 0;
        for (auto& kv : task_index_) {
            Task* t = kv.second;
            if (t->finish_time_ms < 0) continue; // ignore unfinished (shouldn't happen)
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

// --------------------
// Program entry point
// --------------------
int main(int argc, char** argv) {
    auto args  = parse_args(argc, argv);
    auto tasks = args.tasks_path.empty()
               ? default_tasks()
               : load_tasks_csv(args.tasks_path);

    Scheduler sched(args, std::move(tasks));
    sched.run();
    return 0;
}
