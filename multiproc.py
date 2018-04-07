import multiprocessing as mp
import random

from callable import square, cube, tesseract, sct, num_range, process_item, scientists, rand_string
print("number of processors " + str(mp.cpu_count()))

# scientists
pool = mp.Pool()
result = pool.map_async(process_item, scientists)

print(result.get())


# rand_string and queuing
random.seed(123)
# Define an output queue
output = mp.Queue()

# Setup a list of processes that we want to run
processes = [mp.Process(target=rand_string, args=(4, output)) for x in range(10)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]

print(results)

# simple numbers
pool = mp.Pool(mp.cpu_count() - 1)
results = [pool.apply_async( square, args=(n, ) ) for n in range(8)]
output = [r.get() for r in results]
print(output)

pool = mp.Pool(mp.cpu_count() - 1)
results = [pool.apply_async( tesseract, args=(n, ) ) for n in num_range(9)]
output = [r.get() for r in results]
print(output)

pool = mp.Pool(mp.cpu_count() - 1)
results = [pool.apply_async( sct, args=(n, n+1, n+2) ) for n in num_range(9)]
output = [r.get() for r in results]
print(output)
