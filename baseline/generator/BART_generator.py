#! /usr/bin/env python3

# https://towardsdatascience.com/teaching-bart-to-rap-fine-tuning-hugging-faces-bart-model-41749d38f3ef

# imports
from transformers import (BartTokenizer,
                          BartForConditionalGeneration,
                          AdamW,
                          BartConfig)
from torch.utils.data import (DataLoader,
                              TensorDataset,
                              random_split,
                              Dataset)
import pandas as pd
import numpy as np

import torch.nn.functional as F
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import math
import random
import re
import argparse

import operator
import copy
import logging
import psutil

import os
import sys

sys.path.append("../utils")
from data_utils import train_val_test_split_df
from file_utils import mkdir

logging.basicConfig(level=logging.INFO)

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CorpusDataset(pd.DataFrame):
    def __init__(self, path_to_file: str):
        super().__init__(self.get_df(path_to_file))

    @staticmethod
    def get_df(path):
        df = pd.read_csv(path)
        df = df.loc[df["text"] != "\n"]
        return df

# Create a dataloading module as per the PyTorch Lightning Docs
class AnswerGenerationData(Dataset):
    def __init__(self,
                 path_to_df: str = None,
                 df: pd.DataFrame = None,
                 text_column: str = None,
                 corpus: CorpusDataset = None):

        assert operator.xor(path_to_df is None, df is None)
        self.df = self.get_dataset(path_to_df=path_to_df,
                                   df=df,
                                   text_column=text_column,
                                   corpus=corpus)

    @staticmethod
    def get_dataset(path_to_df: str = None,
                    df: pd.DataFrame = None,
                    text_column: str = None,
                    corpus: CorpusDataset = None):

        def get_df(path=None, df=None, text_column=None):
            def parse_ids(ids_str):
                ids_list = [ids[1:-1] for ids in ids_str[1:-1].split(", ")]
                return ids_list

            if path is not None:
                df = pd.read_csv(path)

            columns = ["query",
                       "outline",
                       "text",
                       "paragraphs_id"]
            df = df[["query",
                     "outline",
                     "text_" + text_column,
                     "paragraphs_id"]]
            df = df.loc[df["outline"].map(len) > 0]
            df["text"] = df["text_" + text_column]
            df["paragraphs_id"] = df["paragraphs_id"].apply(parse_ids)
            df = df.loc[df["text"] != "\n"]
            df = df[columns]
            return df

        def get_paragraphs(docs_id, corpus):
            stack = ""
            for index in docs_id:
                stack += corpus[corpus["id"] == index]["text"].item() + "\n"
            stack = stack[:-1]
            return stack

        assert operator.xor(path_to_df is not None, df is not None)

        df = get_df(path_to_df, df, text_column)
        corpus = corpus

        df.rename(columns={"text": "target"}, inplace=True)
        df["source"] = df["paragraphs_id"].apply(lambda ids: get_paragraphs(ids, corpus))
        df = df[["source", "target"]]  #, "outline"]]
        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if type(item) == int:
            return self.df[item:item + 1].to_dict('records')[0]
        elif type(item) == str:
            return self.df[item]

def get_datasets(path_to_file, path_to_corpus, text_column, ):
    corpus = CorpusDataset(path_to_file=path_to_corpus)
    df = pd.read_csv(path_to_file)
    df_train, df_val, df_test = train_val_test_split_df(df)
    del df

    df_train = AnswerGenerationData(df=df_train,
                                    text_column=text_column,
                                    corpus=corpus)
    df_val = AnswerGenerationData(df=df_val,
                                  text_column=text_column,
                                  corpus=corpus)
    df_test = AnswerGenerationData(df=df_test,
                                   text_column=text_column,
                                   corpus=corpus)

    return df_train, df_val, df_test

class Bart(pl.LightningModule):
    # Instantiate the model
    def __init__(self,
                 model_name='facebook/bart-base',
                 train_val_test: tuple = None,
                 freeze_encoder=True,
                 freeze_embeds=True,
                 eval_beams=4,
                 batch_size=32,
                 num_workers=20,
                 learning_rate=1e-5):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.eval_beams = eval_beams
        self.batch_size = batch_size

        self.train_ds, self.val_ds, self.test_ds = train_val_test

        self.freeze_layers()

        if num_workers == -1:
            logging.info(f"Number of CPUs available : {psutil.cpu_count()}.")
            self.num_workers = psutil.cpu_count()
        else:
            self.num_workers = min(num_workers, psutil.cpu_count())

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    @staticmethod
    def freeze_params(model):
        """ Function that takes a model as input (or part of a model) and freezes the layers for faster training
                adapted from finetune.py """
        for layer in model.parameters():
            layer.requires_grade = False

    def freeze_embeds(self):
        """ freeze the positional embedding parameters of the model; adapted from finetune.py """
        self.freeze_params(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            self.freeze_params(d.embed_positions)
            self.freeze_params(d.embed_tokens)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          collate_fn=lambda x: x)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=lambda x: x)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=lambda x: x)

    # Do a forward pass through the model
    def forward(self,
                input_ids,
                **kwargs):
        return self.model(input_ids.to(self.device),
                          **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate)
        return optimizer

    @staticmethod
    def shift_tokens_right(input_ids: torch.Tensor,
                           pad_token_id: torch.Tensor):
        """ Shift input ids one token to the right,
         and wrap the last non pad token (usually <eos>).
                This is taken directly from modeling_bart.py
        """
        prev_output_tokens = input_ids.clone()
        index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        prev_output_tokens[:, 0] = input_ids.gather(1,
                                                    index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        return prev_output_tokens

    def training_step(self, batch, batch_idx):
        loss = 0
        for data in batch:
            input_ids_source, attention_mask_source = self.tokenizer(data["source"],
                                                                     truncation=True,
                                                                     max_length=512,
                                                                     padding="max_length",
                                                                     return_tensors='pt').values()
            input_ids_target, attention_mask_target = self.tokenizer(data["target"],
                                                                     truncation=True,
                                                                     max_length=512,
                                                                     padding="max_length",
                                                                     return_tensors='pt').values()

            # Shift the decoder tokens right (but NOT the tgt_ids)
            decoder_input_ids = self.shift_tokens_right(input_ids_target,
                                                        self.tokenizer.pad_token_id)

            # Run the model and get the logits
            outputs = self(input_ids_source.to(self.device),
                           attention_mask=attention_mask_source.to(self.device),
                           decoder_input_ids=decoder_input_ids.to(self.device),
                           use_cache=False)
            lm_logits = outputs["logits"]
            # Calculate the loss on the un-shifted tokens
            loss += self.loss_fn(lm_logits.view(-1, lm_logits.shape[-1]).to(self.device),
                                 input_ids_target.view(-1).to(self.device))
            self.log('train/loss_step', loss.item(), on_step=True, batch_size=self.batch_size)

        self.log('train/loss_epoch', loss.item(), on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.train_loss = loss
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        logging.info(f'Finishing  epoch {str(self.current_epoch).rjust(5)} - loss : {str(self.train_loss).rjust(15)}')

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for data in batch:
                input_ids_source, attention_mask_source = self.tokenizer(data["source"],
                                                                         truncation=True,
                                                                         max_length=512,
                                                                         padding="max_length",
                                                                         return_tensors='pt').values()
                input_ids_target, attention_mask_target = self.tokenizer(data["target"],
                                                                         truncation=True,
                                                                         max_length=512,
                                                                         padding="max_length",
                                                                         return_tensors='pt').values()

                # Shift the decoder tokens right (but NOT the tgt_ids)
                decoder_input_ids = self.shift_tokens_right(input_ids_target,
                                                            self.tokenizer.pad_token_id)

                # Run the model and get the logits
                outputs = self(input_ids_source.to(self.device),
                               attention_mask=attention_mask_source.to(self.device),
                               decoder_input_ids=decoder_input_ids.to(self.device),
                               use_cache=False)
                lm_logits = outputs["logits"]
                # Calculate the loss on the un-shifted tokens
                loss += self.loss_fn(lm_logits.view(-1, lm_logits.shape[-1]).to(self.device),
                                     input_ids_target.view(-1).to(self.device))
                self.log('Val/loss_step', loss.item(), on_step=True, batch_size=self.batch_size)

            self.log('Val/loss_epoch', loss.item(), on_step=False, on_epoch=True, batch_size=self.batch_size)
            self.val_loss = loss
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        logging.info(f'Validation epoch {str(self.current_epoch).rjust(5)} - loss : {str(self.val_loss).rjust(15)}')

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass

    # # Method that generates text using the BartForConditionalGeneration's generate() method
    # def generate_text(self,
    #                   text,
    #                   eval_beams,
    #                   early_stopping=True,
    #                   max_len=40):
    #     """Function to generate text"""
    #
    #     generated_ids = self.model.generate(
    #         text["input_ids"],
    #         attention_mask=text["attention_mask"],
    #         use_cache=True,
    #         decoder_start_token_id=self.tokenizer.pad_token_id,
    #         num_beams=eval_beams,
    #         max_length=max_len,
    #         early_stopping=early_stopping
    #     )
    #     return [self.tokenizer.decode(w,
    #                                   skip_special_tokens=True,
    #                                   clean_up_tokenization_spaces=True) for w in generated_ids]

def main(text_column="w/o_heading_first_sentence_by_paragraph",
         save_path_checkpoints="./checkpoints",
         save_path_model="./model",
         model_name="BART_generator",
         small=True):

    if small:
        path_to_file_prefix = "../../../data-subset_pre_processed/"
    else:
        path_to_file_prefix = "/users/iris/rserrano/data-set_pre_processed/"

    ds_train, ds_val, ds_test = get_datasets(
        path_to_file=path_to_file_prefix + "fold-1/articles_train_all_ids.csv",
        path_to_corpus=path_to_file_prefix + "fold-1/corpus_train.csv",
        text_column=text_column,
    )

    bart = Bart(train_val_test=(ds_train, ds_val, ds_test))

    logger = pl_loggers.TensorBoardLogger(save_path_checkpoints, name=model_name)
    checkpoint_callback = ModelCheckpoint(monitor="Val/loss_epoch", mode="min", save_top_k=2, every_n_epochs=1)

    trainer = Trainer(logger=logger,
                      precision=32,
                      accelerator="gpu",
                      gpus=-1,
                      strategy='dp',
                      max_epochs=100,
                      callbacks=[checkpoint_callback],
                      log_every_n_steps=1)
    trainer.fit(model=bart)

    # torch.save(dpr, save_path_model)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(save_path_checkpoints="./checkpoints/" + str(sys.argv[1]))
    else:
        main()
