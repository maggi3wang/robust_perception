import multiprocessing
import time

time_start = time.time()
l = multiprocessing.Lock()

def job(num):
    # l.acquire()
    print (num)
    # l.release()
    time.sleep(1)

pool = multiprocessing.Pool(4)

lst = range(40)
for i in lst:
    pool.apply_async(job, [i])

pool.close()
pool.join()

time_end = time.time()
print('total time:', time_end - time_start)