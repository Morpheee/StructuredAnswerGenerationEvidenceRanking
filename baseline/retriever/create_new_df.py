import pandas as pd
import numpy as np
from tqdm import tqdm
import click
import logging
from pprint import pprint

logging.basicConfig(level=logging.INFO)

@click.command()
@click.option("--test-new-path", default=None, help="path to retrieved df, created by dpr_test")
@click.option("--test-old-path", default=None, help="path to old df test")
@click.option("--save-path", default=None, help="path to save file (merged)")
def main(test_new_path,
         test_old_path,
         save_path):
    logging.info(f"Load {test_new_path}")
    df_test_new = pd.read_json(test_new_path)
    logging.info(f"Load {test_old_path}")
    df_test = pd.read_json(test_old_path)
    logging.info(f"Merge left")
    df_test["query"] = df_test["query"].apply(lambda x : x.replace("///"," "))
    pprint(f"df_test_new \n{df_test_new}")
    print("\n"*10)
    pprint(f"df_test \n{df_test}")
    df_test = df_test.merge(df_test_new, how="left", on="query")
    df_test.rename(columns={"id" : "true_id",
                            "retrieved_id" : "id"},
                   inplace=True)
    logging.info(f"Save {save_path}")
    df_test.to_json(save_path, indent=True)
    logging.info("Done.")

if __name__ == "__main__" :
    main()