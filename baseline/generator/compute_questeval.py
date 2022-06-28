import torch
import pandas as pd
import numpy as np
import logging
import click
import sys
import time
from questeval.questeval_metric import QuestEval


logging.basicConfig(level=logging.INFO)

@click.command()
@click.option("--df-output-path", default="", help="path to output file from BART_generator test script")
def main(df_output_path) :
    time_start = time.time()
    logging.info(f"Load {df_output_path}")
    df_output_test = pd.read_json(df_output_path)
    logging.info("Get  questEval model.")
    questeval = QuestEval()
    logging.info("Computing QuestEval score")
    hypothesis = df_output_test["generated_text"].to_list()
    sources = df_output_test["target_text"].to_list()
    score = questeval.corpus_questeval(hypothesis=hypothesis, sources=sources)
    logging.info("Computing over.")
    print("score at example level :", score["ex_level_scores"])
    print("score global :", score["corpus_score"])
    logging.info("DONE.")
    logging.info(f"\t↪ elapsed time : {time.time()-time_start:.2f}s.")
    
    
if __name__ =="__main__":
    main()