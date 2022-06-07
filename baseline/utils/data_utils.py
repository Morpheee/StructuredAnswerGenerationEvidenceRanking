#! /usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split



def train_val_test_split_df(df: pd.DataFrame,
                            train_size=0.6,
                            val_size=0.2,
                            test_size=0.2,
                            print_fn=None):

    assert train_size+val_size+test_size == 1

    df_train, df_rem = train_test_split(df, train_size=train_size)
    df_val, df_test = train_test_split(df_rem, test_size=val_size)

    if print_fn is not None :
        print_fn(f"Samples :\n"
                 f"\t- training   set : {len(df_train)}\n"
                 f"\t- validation set : {len(df_val)}\n"
                 f"\t- test       set : {len(df_test)}")

    return df_train, df_val, df_test
