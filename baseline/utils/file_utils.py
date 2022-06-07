#! /usr/bin/env python3

import os

def mkdir(path: str, model_name: str = None):
    if model_name is not None:
        full_path = os.path.join(path, model_name)
    else:
        full_path = str(path)
    if not os.path.exists(full_path):
        full_path = full_path.split("/")
        for i in range(len(full_path)):
            try:
                os.mkdir("/".join(full_path[:i + 1]))
            except FileExistsError:
                continue