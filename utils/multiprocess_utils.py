import re
import traceback
from multiprocessing import Queue, Process, current_process

is_main_process = not bool(re.match(r'Process-\d+', current_process().name))


def main_process_print(self, *args, sep=' ', end='\n', file=None):
    if is_main_process:
        print(self, *args, sep=sep, end=end, file=file)


def chunked_worker(worker_id, map_func, args, results_queue=None, init_ctx_func=None):
    ctx = init_ctx_func(worker_id) if init_ctx_func is not None else None
    for job_idx, arg in args:
        try:
            if ctx is not None:
                res = map_func(*arg, ctx=ctx)
            else:
                res = map_func(*arg)
            results_queue.put((job_idx, res))
        except:
            traceback.print_exc()
            results_queue.put((job_idx, None))

def chunked_multiprocess_run(map_func, args, num_workers, ordered=True, init_ctx_func=None, q_max_size=1000):
    args = zip(range(len(args)), args)
    args = list(args)
    n_jobs = len(args)
    results_queues = []
    if ordered:
        for i in range(num_workers):
            results_queues.append(Queue(maxsize=q_max_size // num_workers))
    else:
        results_queue = Queue(maxsize=q_max_size)
        for i in range(num_workers):
            results_queues.append(results_queue)
    workers = []
    for i in range(num_workers):
        args_worker = args[i::num_workers]
        p = Process(target=chunked_worker, args=(
            i, map_func, args_worker, results_queues[i], init_ctx_func), daemon=True)
        workers.append(p)
        p.start()
    for n_finished in range(n_jobs):
        results_queue = results_queues[n_finished % num_workers]
        job_idx, res = results_queue.get()
        assert job_idx == n_finished or not ordered, (job_idx, n_finished)
        yield res
    for w in workers:
        w.join()
        w.close()
