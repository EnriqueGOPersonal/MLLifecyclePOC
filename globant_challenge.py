# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 03:15:52 2020

@author: enriq
"""

import pandas as pd
from itertools import combinations , combinations_with_replacement, permutations, product
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="my bar!")
import multiprocessing as mp

c = range(10)

# res = pd.DataFrame(list(product(a,b,b,b,b)))
# res.columns = columns

lets = pd.DataFrame(list(permutations(c)), columns = ["Z", "R", "N", "V", "A", "E", "U", "T", "H", "S"])
lets = lets[lets["S"] > 0]
lets = lets[lets["U"] <= 8]

lets["col1"] = lets.progress_apply(lambda x: x.S*2 + x.H, axis = 1)
lets["N2"] = lets.col1 % 10
lets["res1"] = lets.progress_apply(lambda x: np.floor(x.col1 / 10), axis = 1)

lets = lets[lets["N2"] == lets["N"]]

lets["col2"] = lets..progress_apply(lambda x: x.U*2 + x["T"] + x.res1, axis =1)
lets["R2"] = lets.col2 % 10
lets["res2"] = lets.apply(lambda x: np.floor(x.col2 / 10), axis = 1)

test = lets.head(100)

lets = lets[lets["R2"] == lets["R"]]

lets["col3"] = lets.apply(lambda x: x.N*2 + x["R"] + x.res2, axis =1)
lets["U2"] = lets.col3 % 10
lets["res3"] = lets.apply(lambda x: np.floor(x.col3 / 10), axis = 1)

lets = lets[lets["U2"] == lets["U"]]

lets["col4"] = lets.apply(lambda x: x.A*2 + x["E"] + x.res3, axis =1)
lets["T2"] = lets.col4 % 10
lets["res4"] = lets.apply(lambda x: np.floor(x.col4 / 10), axis = 1)

lets = lets[lets["T2"] == lets["T"]]

lets["col5"] = lets.apply(lambda x: x.V + x["R"] + x["E"] + x.res4, axis =1)
lets["A2"] = lets.col5 % 10
lets["res5"] = lets.apply(lambda x: np.floor(x.col5 / 10), axis = 1)

lets = lets[lets["A2"] == lets["A"]]

lets["col6"] = lets.apply(lambda x: x.res5 + x["U"], axis =1)
lets["S2"] = lets.col6 % 10

lets = lets[lets["S2"] == lets["S"]]

import multiprocessing
import time

start = time.perf_counter()
_name_ = "__main__"

def do_something():
    print('Sleeping for 1 second')
    time.sleep(1)
    print('Done sleeping...')

if _name_ == '__main__':
    p1 = multiprocessing.Process(target=do_something)
    p2 = multiprocessing.Process(target=do_something)
    p1.start()
    p2.start()

    p1.join()
    p2.join()


    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,4)} seconds')