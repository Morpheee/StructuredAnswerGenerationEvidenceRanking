#!/usr/bin/env python3

import pandas as pd
import torch
import sympy
import os
import logging
from tqdm import tqdm
tqdm.pandas(miniters=10000, mininterval=60, maxinterval=600)

logging.basicConfig(level=logging.INFO)

str_nb = [str(j) for j in range(10)]
primes = list(sympy.sieve.primerange(500000, 750000))


def index_to_int(index):
    # index = [str(i) if i in str_nb else str(ord(i) - 97) for i in index]
    # return int(''.join(index))
    return [int(i) for i in index]

def get_factor(index):
    assert primes != []
    factors = []
    for prime in primes:
        if index % prime == 0:
            factors.append(prime)
    for f in factors:
        primes.remove(f)
    return factors


def reduce_index(index, smallest_prime):
    if type(index) == int :
        return str(index % smallest_prime)
    elif type(index) == list :
        return [str(i%smallest_prime) for i in index]


def main(smallest_prime=None):
    if os.path.exists("./ids.json"):
        logging.info("load ids.json")
        ids = pd.read_json("ids.json", dtype={"id": str})
        ids["id"] = ids["id"].progress_apply(int)

        logging.info("sanity check : not any id type != int")
        assert not any(ids["id"].progress_apply(lambda x : type(x) != int))
    else :
        ids = []
        for i in range(5):
            logging.info(f"Load ../../../data-set_pre_processed/fold-{i}/corpus_train.json")
            ids += pd.read_json(f"../../../data-set_pre_processed/fold-{i}/corpus_train.json")["id"].to_list()
        ids = pd.DataFrame({"id": ids})

        logging.info("save to json")
        ids.to_json("ids.json", indent=True)

        logging.info("apply(index_to_int)")
        ids["id"] = ids["id"].progress_apply(index_to_int)

        logging.info("sanity check")
        assert not any(ids["id"].progress_apply(lambda x : type(x) != int))

        logging.info("save to json")
        ids["id"] = ids["id"].apply(str)
        ids.to_json("ids.json", indent=True)

    if smallest_prime is None:
        logging.info("apply(get_factor)")
        ids["id"] = ids["id"].progress_apply(get_factor)

        smallest_prime = primes[0]

        logging.info(f"smallest_prime : {smallest_prime}")
        ids["id"] = ids["id"].progress_apply(lambda x: reduce_index(x, smallest_prime))

    logging.info(f"sanity check : all id % smallest_prime != 0")
    sanity_check = ids["id"].progress_apply(lambda x : x%smallest_prime != 0)
    assert all(sanity_check)

    files = []
    for file_name in ["articles_train.json",
                      "corpus_train.json",
                      "sections_train.json"] :
        files += [f"../../../data-set_pre_processed/fold-{i}/{file_name}" for i in range(5)]

    for file in files :
        logging.info(f"Processing {file}...")
        df = pd.read_json(file)
        for col in df.columns :
            if "id" in col :
                logging.info(f"Current column : {col}")
                df[col] = df[col].progress_apply(index_to_int)
                df[col] = df[col].progress_apply(lambda x : reduce_index(x, smallest_prime))
                assert all(df[col].apply(lambda x : x!=0))
            else :
                continue
        df.to_json(file+".new", indent=True)


if __name__ == "__main__":
    main(618833)
