import torch
import numpy as np
from tqdm import tqdm
import os
import click
import pandas as pd
import logging
import re


@click.command()
@click.option("--directory", default="", required=True,
              help="dense_tensors_directory")
def main(directory):
    tensors = []
    for file in os.listdir(directory) :
        if file.startswith("dense_tensor.start") :
            print(file)
            start, end = re.findall(r"\d+", file)
            tensors.append([int(start), int(end), torch.load(os.path.join(directory,file))])
    dense = torch.zeros(size=(max(t[1] for t in tensors),768), dtype=torch.float64)
    for t in tensors :
        a = t[0]
        b = t[1]
        dense[a:b] = t[2][a:b]
    torch.save(dense, os.path.join(directory, "dense.ckpt"))
    for file in os.listdir(directory) :
        if file.startswith("dense_tensor.start") :
            print(f"del {file}")
            os.remove(os.path.join(directory,file))


if __name__ == "__main__":
    main()