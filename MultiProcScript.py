import time
import os
import multiprocessing
from tqdm import tqdm
import random

def myfunc():
    print('Performing some tasks before fork')


    iterations = 12
    processes = 5
    counter = 0
    semaphore = multiprocessing.Semaphore(processes)
    progress = multiprocessing.Value('i', -1)
    processes_list = []

    pbar = tqdm(total=iterations, leave=True)
    #with tqdm(total=0) as pbar:
    while counter < iterations:
        for _ in range(processes):
            if counter >= iterations:
                break

            semaphore.acquire()
            pid = os.fork()

            if pid==0:
                #print("Starting child porcess: iteration {}".format(counter+1))
                time.sleep(random.randint(5,15))
                #print("Child process terminating")
                semaphore.release()
                with progress.get_lock():
                    progress.value+=1
                    pbar.n = progress.value
                    pbar.last_print_n = progress.value
                    pbar.update()
                os._exit(0)
            else:
                processes_list.append(pid)
                counter+=1
                time.sleep(1)


    while processes_list:
        pid, exit_code = os.wait()
        if pid==0:
            time.sleep(1)
        else:
            processes_list.remove(pid)

    pbar.close()
    print("All children terminated, finishing up")

    return



myfunc()
