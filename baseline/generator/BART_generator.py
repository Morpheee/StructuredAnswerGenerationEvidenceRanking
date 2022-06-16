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
from tqdm import tqdm

import time
import os
import sys

sys.path.append("../utils")
from data_utils import train_val_test_split_df
from file_utils import mkdir
from metrics import rouge, bleu, meteor, bert_score

import click

logging.basicConfig(level=logging.INFO)

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
rng = np.random.RandomState(seed=42)


def get_datasets(
        train_ds,
        train_corpus,
        val_ds,
        val_corpus,
        test_ds,
        test_corpus,
        text_column
):
    df_train = pd.read_json(train_ds)
    corpus_train = CorpusDataset(path_to_file=train_corpus, text_column=text_column)
    df_val = pd.read_json(val_ds).sample(int(len(df_train) * .1))
    corpus_val = CorpusDataset(path_to_file=val_corpus, text_column=text_column)
    df_test = pd.read_json(test_ds)
    corpus_test = CorpusDataset(path_to_file=test_corpus, text_column=text_column)

    df_train = AnswerGenerationData(df=df_train,
                                    text_column=text_column,
                                    corpus=corpus_train)
    df_val = AnswerGenerationData(df=df_val,
                                  text_column=text_column,
                                  corpus=corpus_val)
    df_test = AnswerGenerationData(df=df_test,
                                   text_column=text_column,
                                   corpus=corpus_test)

    return df_train, df_val, df_test


class CorpusDataset(pd.DataFrame):
    def __init__(self, path_to_file: str, text_column: str = None):
        super().__init__(self.get_df(path_to_file, text_column))

    @staticmethod
    def get_df(path, text_column):
        if path.endswith(".json"):
            df = pd.read_json(path, dtype={"id": str})
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        if "all" in text_column:
            df.rename(columns={"all_passage": "text"}, inplace=True)
        elif "first" in text_column:
            df.rename(columns={"first_sentence": "text"}, inplace=True)
        df = df.loc[df["text"] != "\n"]
        df["id"] = df["id"].apply(int)
        return df


class AnswerGenerationData(Dataset):
    def __init__(self,
                 path_to_df: str = None,
                 df: pd.DataFrame = None,
                 text_column: str = None,
                 corpus: CorpusDataset = None):

        assert operator.xor(path_to_df is None, df is None)
        self.df = self.get_dataset(path_to_df=path_to_df,
                                   df=df,
                                   text_column=text_column)
        self.corpus = corpus

    @staticmethod
    def get_dataset(path_to_df: str = None,
                    df: pd.DataFrame = None,
                    text_column: str = None):

        def parse_ids(ids_raw):
            if type(ids_raw) == str:
                ids_list = [ids[1:-1] for ids in ids_raw[1:-1].split(", ")]
                return ids_list
            elif type(ids_raw) == list:
                ids_int = [int(i) for i in ids_raw]
                return ids_int

        assert operator.xor(path_to_df is not None, df is not None)

        if path_to_df is not None:
            # df = pd.read_csv(path)
            df = pd.read_json(path_to_df)
        columns = ["query",
                   "outline",
                   "text",
                   "id"]
        df = df[["query",
                 "outline",
                 "text_" + text_column,
                 "id"]]
        df = df.loc[df["outline"].map(len) > 0]
        df["text"] = df["text_" + text_column]
        df["id"] = df["id"].apply(parse_ids)
        df = df[columns]
        df = df.loc[df["text"] != "\n"]

        df.rename(columns={"text": "target"}, inplace=True)
        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if type(item) == int:
            row = pd.DataFrame(self.df.loc[item:item])
            row["source"] = row[["query", "id"]].apply(lambda row: self.get_paragraphs(row[0], row[1]), axis=1)
            row = row[["source", "target"]]
            return row.to_dict('records')[0]
        elif type(item) == str:
            return self.df[item]

    def get_paragraphs(self, query, docs_id):
        stack = query + "\n"
        rng.shuffle(docs_id)
        for index in docs_id:
            stack += self.corpus[self.corpus["id"] == index]["text"].item() + "\n"
        stack = stack[:-1]
        return stack


class Bart(pl.LightningModule):
    # Instantiate the model
    def __init__(self,
                 model_name='facebook/bart-base',
                 train_val_test: tuple = None,
                 freeze_params=-1,
                 eval_beams=4,
                 batch_size=4,
                 batch_size_test=1,
                 num_workers=5,
                 learning_rate=1e-5):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.eval_beams = eval_beams
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test

        if train_val_test is not None:
            self.train_ds, self.val_ds, self.test_ds = train_val_test
        else:
            self.train_ds = None
            self.val_ds = None
            self.test_ds = None

        self.freeze_params = freeze_params

        if num_workers == -1:
            logging.info(f"Number of CPUs available : {psutil.cpu_count()}.")
            self.num_workers = psutil.cpu_count()
        else:
            self.num_workers = min(num_workers, psutil.cpu_count())

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        self.metrics = {"bleu": [],
                        "rouge1_precision": [],
                        "rouge1_recall": [],
                        "rouge1_f1": [],
                        "rougeL_precision": [],
                        "rougeL_recall": [],
                        "rougeL_f1": [],
                        "bert_scoreAVG_precision": [],
                        "bert_scoreAVG_recall": [],
                        "bert_scoreAVG_f1": [],
                        "meteor": [],
                        "QuestEval": []}

    def freeze_layers(self):
        if self.freeze_params == -1:
            self.freeze_last_layers()
        else:
            num_layers = sum(1 for _ in self.model.parameters())
            for parameters in list(self.model.parameters())[:int(self.freeze_params * num_layers)]:
                parameters.requires_grad = False
        self.freeze_embeds()

    def freeze_last_layers(self):
        nb_layer = []
        for name, _ in self.model.named_parameters():
            try:
                nb_layer.append(int(re.findall(r"layer\.\d+", name)[0].split(".")[-1]))
            except IndexError:
                continue
        nb_layer = max(set(nb_layer))
        for name, param in self.model.named_parameters():
            if f'layer.{nb_layer}' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def freeze_embeds(self):
        """ freeze the positional embedding parameters of the model; adapted from finetune.py """
        for layer in self.model.model.shared.parameters():
            layer.requires_grade = False
        for d in [self.model.model.encoder, self.model.model.decoder]:
            for layer in d.embed_positions.parameters():
                layer.requires_grade = False
            for layer in d.embed_tokens.parameters():
                layer.requires_grade = False

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
                          batch_size=self.batch_size_test,
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
        self.model.eval()

        with torch.no_grad():
            output = []
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
                generated_ids = self.model.generate(
                    input_ids_source.to(self.device),
                    attention_mask=attention_mask_source.to(self.device),
                    use_cache=True,
                    decoder_start_token_id=self.tokenizer.pad_token_id,
                    num_beams=4,
                    max_length=512,
                    early_stopping=True
                )
                # generated_text = ''.join(
                #     [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                #      for w in generated_ids])
                output.append([input_ids_target, generated_ids])

            return output

    #
    def test_epoch_end(self, outputs):
        print("\n")
        inp_ids_target = []
        gen_ids = []
        for output in outputs:
            for input_ids_target, generated_ids in tqdm(output, miniters=100):
                self.metrics["bleu"].append(bleu(generated_ids, input_ids_target)["score"])
                rouge_comp = rouge(generated_ids, input_ids_target)
                # bert_score_comp = bert_score(generated_ids, input_ids_target)
                self.metrics["rouge1_precision"].append(rouge_comp["rouge1"].mid.precision)
                self.metrics["rouge1_recall"].append(rouge_comp["rouge1"].mid.recall)
                self.metrics["rouge1_f1"].append(rouge_comp["rouge1"].mid.fmeasure)
                self.metrics["rougeL_precision"].append(rouge_comp["rougeL"].mid.precision)
                self.metrics["rougeL_recall"].append(rouge_comp["rougeL"].mid.recall)
                self.metrics["rougeL_f1"].append(rouge_comp["rougeL"].mid.fmeasure)
                # self.metrics["bert_scoreAVG_precision"].append(np.mean(bert_score_comp["precision"]))
                # self.metrics["bert_scoreAVG_recall"].append(np.mean(bert_score_comp["recall"]))
                # self.metrics["bert_scoreAVG_f1"].append(np.mean(bert_score_comp["f1"]))
                inp_ids_target.append(input_ids_target)
                gen_ids.append(generated_ids)
        self.metrics["meteor"].append(meteor(inp_ids_target, gen_ids)["meteor"])
        # self.metrics["QuestEval"] =
        print("\n" * 5)
        for k, v in self.metrics.items():
            if type(v) == list:
                print(f"{str(k).ljust(12)} : {v}")
            else:
                print(f"{str(k).ljust(12)} : {v * 100:.2f}")
        print("\n" * 5)
        for k, v in self.metrics.items():
            if type(v) is list:
                if len(v) > 0:
                    self.metrics[k] = np.mean(v)
        for k, v in self.metrics.items():
            if type(v) == list:
                print(f"{str(k).ljust(12)} : {v}")
            else:
                print(f"{str(k).ljust(12)} : {v * 100:.2f}")


@click.command()
@click.option("--checkpoints-suffix", default="",
              help="suffix of checkpoints save path")
@click.option("--load-from-checkpoint", default=None,
              help="if already computed, path to tensor containing attention masks of corpus (validation step")
@click.option("--text-column", default="w_heading_first_sentence",
              help="text column to keep (w or w/o heading // first_sentence or all_passage")
@click.option("--train-ds", default="fold-0/articles_train.json",
              help="training dataset.")
@click.option("--train-corpus", default="fold-0/corpus_train.json",
              help="training corpus.")
@click.option("--val-ds", default="fold-1/articles_train.json",
              help="training dataset.")
@click.option("--val-corpus", default="fold-1/corpus_train.json",
              help="val corpus.")
@click.option("--test-ds", default="test/articles_test.json",
              help="test dataset.")
@click.option("--test-corpus", default="test/corpus_test.json",
              help="test corpus.")
def main(
        text_column,
        checkpoints_suffix,
        load_from_checkpoint,
        train_ds,
        train_corpus,
        val_ds,
        val_corpus,
        test_ds,
        test_corpus,
        save_path_checkpoints="./checkpoints",
        model_name="BART_generator",
        small=False
):
    start_time = time.time()

    logging.info(" " * 35 + "↪ elapsed time : "
                            f"{int((time.time() - start_time) // 60)}min "
                            f"{(time.time() - start_time) % 60:.2f}s.")
    logging.info(f"Get datasets")

    if small:
        path_to_file_prefix = "../../../data-subset_pre_processed/"
    else:
        path_to_file_prefix = "../../../data-set_pre_processed/"

    ds_train, ds_val, ds_test = get_datasets(
        train_ds=os.path.join(path_to_file_prefix, train_ds),
        train_corpus=os.path.join(path_to_file_prefix, train_corpus),
        val_ds=os.path.join(path_to_file_prefix, val_ds),
        val_corpus=os.path.join(path_to_file_prefix, val_corpus),
        test_ds=os.path.join(path_to_file_prefix, test_ds),
        test_corpus=os.path.join(path_to_file_prefix, test_corpus),
        text_column=text_column
    )

    logging.info(" " * 35 + "↪ elapsed time : "
                            f"{int((time.time() - start_time) // 60)}min "
                            f"{(time.time() - start_time) % 60:.2f}s.")

    if checkpoints_suffix:
        save_path_checkpoints = os.path.join(save_path_checkpoints, checkpoints_suffix)
    logging.info(f"Make logger and checkpoint's folders : '{save_path_checkpoints}'")
    mkdir(save_path_checkpoints, model_name)

    logger = pl_loggers.TensorBoardLogger(save_path_checkpoints, name=model_name)
    checkpoint_callback = ModelCheckpoint(monitor="Val/loss_epoch",
                                          mode="min",
                                          save_last=True,
                                          save_top_k=5,
                                          every_n_epochs=1)

    logging.info(" " * 35 + "↪ elapsed time : "
                            f"{int((time.time() - start_time) // 60)}min "
                            f"{(time.time() - start_time) % 60:.2f}s.")
    logging.info(f"Create Trainer")

    if torch.cuda.is_available():
        trainer = Trainer(logger=logger,
                          precision=32,
                          accelerator="gpu",
                          gpus=-1,
                          strategy='dp',
                          max_epochs=100,
                          callbacks=[checkpoint_callback],
                          log_every_n_steps=1,
                          progress_bar_refresh_rate=100)
    else:
        trainer = Trainer(logger=logger,
                          precision=32,
                          accelerator="cpu",
                          max_epochs=50,
                          callbacks=[checkpoint_callback],
                          log_every_n_steps=1,
                          progress_bar_refresh_rate=100)

    logging.info(" " * 35 + "↪ elapsed time : "
                            f"{int((time.time() - start_time) // 60)}min "
                            f"{(time.time() - start_time) % 60:.2f}s.")

    torch.cuda.empty_cache()
    if load_from_checkpoint:
        logging.info("load model from checkpoint : bart()")
        bart = Bart.load_from_checkpoint(load_from_checkpoint,
                                         train_val_test=(ds_train, ds_val, ds_test))
        logging.info(" " * 35 + "↪ elapsed time : "
                                f"{int((time.time() - start_time) // 60)}min "
                                f"{(time.time() - start_time) % 60:.2f}s.")
    else:
        logging.info("make model : bart()")
        bart = Bart(train_val_test=(ds_train, ds_val, ds_test))
        logging.info(" " * 35 + "↪ elapsed time : "
                                f"{int((time.time() - start_time) // 60)}min "
                                f"{(time.time() - start_time) % 60:.2f}s.")

        logging.info(f"Fit trainer")
        trainer.fit(model=bart)
        logging.info(" " * 35 + "↪ elapsed time : "
                                f"{int((time.time() - start_time) // 60)}min "
                                f"{(time.time() - start_time) % 60:.2f}s.")

    del bart.train_ds
    del bart.val_ds

    logging.info("Test model : bart()")
    trainer.test(model=bart)
    logging.info(" " * 35 + "↪ elapsed time : "
                            f"{int((time.time() - start_time) // 60)}min "
                            f"{(time.time() - start_time) % 60:.2f}s.")

    logging.info(f"DONE.".ljust(50) + f";\telapsed time : " +
                 f"{int((time.time() - start_time) // 60)}min " +
                 f"{(time.time() - start_time) % 60:.2f}s.")


if __name__ == "__main__":
    main()
