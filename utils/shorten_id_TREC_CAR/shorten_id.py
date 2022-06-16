#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import sympy
import os
import logging
import multiprocessing
from joblib import Parallel, delayed
import sys
sys.path.append("../../..")
from trec_car_tools.python3.trec_car.read_data_own import iter_pages, iter_paragraphs
from tqdm import tqdm
tqdm.pandas(miniters=100000, mininterval=60, maxinterval=600)

logging.basicConfig(level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



logging.info("Load all ids...")
ids = []
for i in tqdm(list(range(5))) :
    with open(f"/projets/iris/CORPUS/DOCS/TREC-CAR-Y1/train/train.fold{i}.cbor.paragraphs", "rb") as file :
        for paragraph in iter_paragraphs(file) :
            ids.append(int(paragraph.para_id))
ids = np.array(ids)

if os.path.exists("./primes.npy") :
    primes = np.load("./primes.npy")
else :
    logging.info("Generate primes...")
    # primes = []
    # for i in range(100000) :
    #     primes.append(number.getPrime(8))
    primes = list(sympy.sieve.primerange(100000000,147000000))
    primes = np.array(list(primes))
    logging.info(f"{len(primes)} primes found.")
    np.save("./primes.npy", primes)
    logging.info(f"primes saves as ./primes.npy")

logging.info("Test all possibilities...")

def test_all(primes_list):
    for prime in tqdm(primes_list,
                      miniters=10000,
                      maxinterval=3600) :
        remainder = ids % prime
        if any(remainder == 0) :
            continue
        elif len(set(remainder)) < len(ids) :
            continue
        else :
            smallest_prime = int(prime)
            logging.info(f"smallest_prime : {smallest_prime}")
            break

step = 100000
num_cores = multiprocessing.cpu_count()
inputs = tqdm([primes[i:i+step] for i in range(0,len(primes),step)])
logging.info(f"len inputs : {len(inputs)}")

if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores)(delayed(test_all)(p) for p in inputs)






# def get_primes(lower_bound=100, upper_bound=1000, step=10000) :
#     for i in range(lower_bound,upper_bound) :
#         yield list(sympy.sieve.primerange(i*step,(i+1)*step))


# ids = []
# for i in tqdm(range(5)):
#     logging.info(f"Load ../../../data-set_pre_processed/fold-{i}/corpus_train.json")
#     ids += pd.read_json(f"../../../data-set_pre_processed/fold-{i}/corpus_train.json",
#                         dtype={'id': str})["id"].to_list()
# ids += pd.read_json(f"../../../data-set_pre_processed/test/corpus_test.json",
#                         dtype={'id': str})["id"].to_list()

# ids = [int(i) for i in ids]
# logging.info(f"len(ids) = {len(ids)}")

# abort = False
# for i in tqdm(ids) :
#     if i % 2147483647 == 0 :
#         abort = True
#         print(i)

# assert abort == False
# logging.info(f"{2147483647} is not a factor of any index")
# prime = 2147483647

# del ids

# smallest_prime=None
# if smallest_prime is None:
#     primes_gen = get_primes()
#     smallest_prime = -1
#     j = 0
#     for primes in primes_gen :
#         logging.info(f"primes : {primes[0]} ---> {primes[-1]}")
#         if j < 0 :
#             break
#         j += 1
#         for prime in tqdm(primes) :
#             remainder = ids % prime
#             if any(remainder == 0) :
#                 continue
#             else :
#                 smallest_prime = int(prime)
#                 logging.info(f"smallest_prime : {smallest_prime}")
#                 j = -1
#                 break
#         if j < 0 :
#             break

# if smallest_prime == -1 :
#     exit(1)

# files_name = []
# for suffix_name in [
#                   "corpus_train.json",
#                   "articles_train.json",
#                   "sections_train.json"
#                   ] :
#     files_name += [f"../../../data-set_pre_processed/fold-{i}/{suffix_name}" for i in range(5)]
# files_name += [f"../../../data-set_pre_processed/test/corpus_test.json",
#               f"../../../data-set_pre_processed/test/articles_test.json",
#               f"../../../data-set_pre_processed/test/sections_test.json"]
# files_name = [file_name.replace("data-set", "data-subset") for file_name in files_name] + files_name

# for file_name in files_name :
#     logging.info(f"Processing {file_name}...")
#     try :
#         df = pd.read_json(file_name)
#     except FileNotFoundError:
#         logging.warning("File Not Found")
#         continue
#     if "corpus" in file_name :
#         df["new_id"] = df["id"].apply(lambda x : int(x) % prime)
#         logging.info(r"sanity check : all id % prime != 0")
#         assert all(df["new_id"] != 0)
#     else :
#         df["new_id"] = df["id"].apply(lambda x_list : [int(x) % prime for x in x_list])
#         logging.info(r"sanity check : all id % prime != 0")
#         assert all(df["new_id"].apply(lambda x_list : all([x != 0 for x in x_list])))
#     df["id"] = df["new_id"]
#     df = df.drop(["new_id"], axis=1)
#     logging.info(f"Sanity check passed.")
#     logging.info(f"save file as {file_name}.short.json.")
#     df.to_json(file_name+".short.json", indent=True)
#     logging.info("saved.\n\n")
# logging.info(f"DONE.\n\n"+"="*50+"\n\n")

# logging.info(f"No error, thus moving all tmp file to real file.")
# for file_name in files_name :
#     logging.info(f"Processing {file_name}...")
#     df = pd.read_json(file_name + ".new.json")
#     df["id"] = df["new_id"]
#     df = df.drop(["new_id"], axis=1)
#     df.to_json(file_name.replace(".json","_short.json"), indent=True)
#     sys.path.remove(file_name + ".new.json")
#     logging.info(f"saved.\n")
# logging.info(f"DONE.\n\n")