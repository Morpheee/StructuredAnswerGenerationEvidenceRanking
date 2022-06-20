import torch
import pandas as pd
import numpy as np
import logging
import click
import sys
sys.path.append("../utils")
from data_utils import train_val_test_split_df
from file_utils import mkdir
from metrics import rouge, bleu, meteor, bert_score

logging.basicConfig(level=logging.INFO)

@click.command()
@click.option("--df-output-path", default="", help="path to output file from BART_generator test script")
def main(df_output_path) :
    logging.info(f"Load {df_output_path}")
    df_output_test = pd.read_json(df_output_path)
    logging.info("Computing BertScore...")
    score = bert_score(df_output_test["generated_text"].tolist(),df_output_test["source_text"].tolist())
    logging.info(score)
    for k,v in score.items():
        if type(v) is list :
            logging.info(f"{str(k).ljust(15)} : {np.mean(v)*100:.4f}")
        else :
            logging.info(f"{str(k).ljust(15)} : {v}")
    logging.info("Done.")


if __name__ =="__main__":
    main()