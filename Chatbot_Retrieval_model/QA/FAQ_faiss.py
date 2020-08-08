# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   FAQ_faiss.py
 
@Time    :   2020/6/17 4:29 下午
 
@Desc    :   faiss 检索
 
"""

import numpy as np
import faiss
import time


# d = 64                           # dimension
# nb = 100000                      # database size
# nq = 10000                       # nb of queries
# np.random.seed(1234)             # make reproducible
# xb = np.random.random((nb, d)).astype('float32')
# xb[:, 0] += np.arange(nb) / 1000.
# xq = np.random.random((nq, d)).astype('float32')
# xq[:, 0] += np.arange(nq) / 1000.

# index = faiss.IndexFlatL2(d)
# print(index.is_trained)
# index.add(xb)
# print(index.ntotal)

# k = 4                          # we want to see 4 nearest neighbors
# D, I = index.search(xq, k)     # actual search



# 使用gpu
# ngpus = faiss.get_num_gpus()
# print("Number of GPUs:", ngpus)
#
# cpu_index = faiss.IndexFlatL2(d)
# gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
#
# gpu_index.add(xb)
#
# k = 4
# start_time = time.time()
# D, I = gpu_index.search(xq, k)
# end_time = time.time()
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])
# print('cost time is {}'.format(end_time-start_time))


# nlist = 100                       #聚类中心的个数
# quantizer = faiss.IndexFlatL2(d)  # the other index
# index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
#        # here we specify METRIC_L2, by default it performs inner-product search
# assert not index.is_trained
# index.train(xb)
# assert index.is_trained
#
# index.add(xb)                  # add may be a bit slower as well
# start_time1 = time.time()
# D, I = index.search(xq, k)     # actual search
# end_time1 = time.time()
# print(I[-5:])                  # neighbors of the 5 last queries
# print('Cost time 1 is {}'.format(end_time1 - start_time1))
# index.nprobe = 10              # default nprobe is 1, try a few more
# start_time2 = time.time()
# D, I = index.search(xq, k)
# end_time2 = time.time()
# print(I[-5:])                  # neighbors of the 5 last queries
# print('Cost 2 time is {}'.format(end_time2 - start_time2))

d = 512
nb = 3000000
nq = 100
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000

quantizer = faiss.IndexFlatL2(d)
nlist = 100
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)      # 聚类

gpu_index = faiss.index_cpu_to_all_gpus(index)
print(gpu_index.is_trained)
gpu_index.train(xb)
print(gpu_index.is_trained)

gpu_index.add(xb)
gpu_index.nprobe = 10
start_time = time.time()

D, gt_nns = gpu_index.search(xq, 1)
end_time = time.time()
print('Cost time is {}'.format(end_time - start_time))