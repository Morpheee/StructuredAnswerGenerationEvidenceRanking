#! /usr/bin/env python3
# https://github.com/huggingface/transformers/blob/main/examples/research_projects/rag-end2end-retriever/finetune_rag.py
import numpy as np
from sys import getsizeof
import os
from transformers import T5Tokenizer
from icecream import ic
import time
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
import pandas as pd
import logging
import random
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

logging.basicConfig(level=logging.INFO)


class CorpusDataset(pd.DataFrame):
    def __init__(self, path_to_file: str, context_encoder):
        super().__init__(self.get_df(path_to_file, context_encoder))

    def get_df(self, path, context_encoder):
        df = pd.read_csv(path)
        df = df.loc[df["text"] != "\n"]
        df = context_encoder(df)
        return df


class QueryDataset(Dataset):
    def __init__(self, path_to_file: str, text_column: str, query_encoder, corpus, nb_irrelevant=1):
        self.df = self.get_df(path_to_file, text_column, query_encoder)
        self.corpus = corpus
        self.nb_irrelevant = nb_irrelevant
        self.count_doc = [0 for _ in range(len(self.df))]

    def get_df(self, path, text_column=None, query_encoder=None):
        def parse_ids(ids_str):
            ids_list = [id[1:-1] for id in ids_str[1:-1].split(", ")]
            return ids_list

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
        df = query_encoder(df)
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        docs_id = row["paragraphs_id"]
        query = {"query": row["query"],
                 "input_ids": row["input_ids"],
                 "attention_mask": row["attention_mask"]}
        positive = self.get_positive(docs_id[self.count_doc[item]])
        negative = self.get_negative(docs_id)
        element = {"query": query,
                   "positive": positive,
                   "negative": negative}
        self.count_doc[item] = (self.count_doc[item] + 1) % len(docs_id)
        return element

    def get_positive(self, doc_id):
        positive = {"doc_id": doc_id}
        row = self.corpus[self.corpus["id"] == doc_id]
        positive["input_ids"] = row["input_ids"].values[0]
        positive["attention_mask"] = row["attention_mask"].values[0]
        return positive

    def get_negative(self, docs_id_positive):
        negatives_documents = []
        for _ in range(self.nb_irrelevant):
            item_random = np.random.randint(0, len(self.corpus))
            row_negative = self.corpus.iloc[item_random]
            while row_negative["id"] in docs_id_positive:
                item_random = np.random.randint(0, len(self.corpus))
                row_negative = self.corpus.iloc[item_random]
            negative = {"doc_id": row_negative["id"],
                        "input_ids": row_negative["input_ids"],
                        "attention_mask": row_negative["attention_mask"]}
            negatives_documents.append(negative)
        return negatives_documents


class DPR(pl.LightningModule):
    """
    Implementation of the DPR module :
    Encode all documents (contexts), and query with different BERT encoders.
    Similarity measure with dot product.
    """

    def __init__(self,
                 query_model_name: str,
                 context_model_name: str,
                 train_batch_size=2,
                 val_batch_size=2,
                 learning_rate=1e-5):
        super().__init__()
        logging.info("query_tokenizer\n\n\n\n")
        self.query_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(query_model_name)
        logging.info("context_tokenizer\n\n\n\n")
        self.context_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(context_model_name)
        logging.info("query_model\n\n\n\n")
        self.query_model = DPRQuestionEncoder.from_pretrained(query_model_name)
        logging.info("context_model\n\n\n\n")
        self.context_model = DPRContextEncoder.from_pretrained(context_model_name)

        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.train_ds, self.val_ds, self.test_ds = None, None, None

        # self.loss_fn = nn.NLLLoss()
        self.loss_fn = self.get_loss_fn()

    def get_sets(self, train_val_test):
        self.train_ds, self.val_ds, self.test_ds = train_val_test

    def encode_queries(self, queries):
        qry_enc = lambda qry: self.query_tokenizer(qry, truncation=True,
                                                   max_length=512,
                                                   padding="max_length",
                                                   return_tensors="pt")

        queries_encoding = pd.DataFrame(queries["text"].apply(qry_enc).tolist())
        queries["input_ids"] = queries_encoding["input_ids"]
        queries["attention_mask"] = queries_encoding["attention_mask"]
        return queries

    def encode_contexts(self, contexts: pd.DataFrame):
        ctx_enc = lambda ctx: self.context_tokenizer(ctx, truncation=True,
                                                     max_length=512,
                                                     padding="max_length",
                                                     return_tensors='pt')
        contexts_encoding = pd.DataFrame(contexts["text"].apply(ctx_enc).tolist())
        contexts["input_ids"] = contexts_encoding["input_ids"]
        contexts["attention_mask"] = contexts_encoding["attention_mask"]
        return contexts

    def decode_contexts(self, contexts_encodings: list):
        contexts = [self.context_tokenizer.decode(c) for c in contexts_encodings]
        return contexts

    def get_dense_query(self, query):
        dense_query = self.query_model(input_ids=query["input_ids"],
                                       attention_mask=query["attention_mask"])["pooler_output"]
        return dense_query

    def get_dense_contexts(self, contexts):
        dense_embeddings = []
        for context in contexts:
            embedding = self.context_model(input_ids=context["input_ids"],
                                           attention_mask=context["attention_mask"])
            dense_embeddings.append(embedding["pooler_output"])
        return torch.cat(dense_embeddings)

    def dot_product(self, query, contexts):
        sim = query.squeeze().matmul(contexts.T)
        return sim.squeeze()

    def train_dataloader(self):
        return DataLoader(copy.deepcopy(self.train_ds),
                          batch_size=self.train_batch_size,
                          num_workers=4,
                          shuffle=True,
                          collate_fn=lambda x: x)

    def val_dataloader(self):
        return DataLoader(copy.deepcopy(self.val_ds),
                          batch_size=self.val_batch_size,
                          num_workers=4,
                          shuffle=True,
                          collate_fn=lambda x: x)

    def test_dataloader(self):
        return DataLoader(copy.deepcopy(self.test_ds),
                          batch_size=self.val_batch_size,
                          num_workers=4,
                          shuffle=True,
                          collate_fn=lambda x: x)

    def get_loss_fn(self):
        """negative log likelihood from DPR paper."""

        def loss_fn(similarity):
            return -torch.log(similarity[0].exp() / similarity.exp().sum())

        return loss_fn

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        :param query:
        :param return_contexts:
        :param k:
        :return:
        """
        self.context_model.train()
        self.query_model.train()
        loss = 0
        for data in batch:
            query_dense = self.get_dense_query(data["query"])
            contexts_dense = self.get_dense_contexts([data["positive"], *data["negative"]])
            similarity = self.dot_product(query_dense, contexts_dense)
            loss += self.loss_fn(similarity)

        loss /= len(batch)
        self.log('train/loss_step', loss.item(), on_step=True)
        self.log('train/loss_epoch', loss.item(), on_step=False, on_epoch=True)

        return loss

    def training_epoch_end(self, outputs):
        logging.info('Finishing epoch ', self.current_epoch)

    def validation_step(self, batch, batch_idx):
        self.context_model.eval()
        self.query_model.eval()
        loss = 0
        with torch.no_grad():
            for data in batch:
                self.context_model.train()
                self.query_model.train()
                query_dense = self.get_dense_query(data["query"])
                contexts_dense = self.get_dense_contexts([data["positive"], *data["negative"]])
                similarity = self.dot_product(query_dense, contexts_dense)
                loss += self.loss_fn(similarity)

            loss /= len(batch)
            self.log('Val/loss_step', loss.item(), on_step=True)
            self.log('Val/loss_epoch', loss.item(), on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs):
        logging.info('Validation epoch ', self.current_epoch)


def main(text_column="w/o_heading_first_sentence_by_paragraph"):
    start_time = time.time()
    logging.info("make model : dpr()")
    dpr = DPR(context_model_name="facebook/dpr-ctx_encoder-single-nq-base",
              query_model_name="facebook/dpr-question_encoder-single-nq-base")

    logging.info("elapsed time : "
                 f"{int((time.time() - start_time) // 60)}min "
                 f"{(time.time() - start_time) % 60:.2f}s.")
    logging.info(f"Get corpus")

    corpus_train = CorpusDataset(
        # path_to_file="/users/iris/rserrano/data-set_pre_processed/fold-1/corpus_train.csv",
        path_to_file="../../../data-subset_pre_processed/fold-1/corpus_train.csv",
        context_encoder=dpr.encode_contexts
    )
    corpus_val = CorpusDataset(
        # path_to_file="/users/iris/rserrano/data-set_pre_processed/fold-2/corpus_train.csv",
        path_to_file="../../../data-subset_pre_processed/fold-2/corpus_train.csv",
        context_encoder=dpr.encode_contexts
    )
    corpus_test = CorpusDataset(
        # path_to_file="/users/iris/rserrano/data-set_pre_processed/fold-3/corpus_train.csv",
        path_to_file="../../../data-subset_pre_processed/fold-3/corpus_train.csv",
        context_encoder=dpr.encode_contexts
    )

    logging.info("elapsed time : "
                 f"{int((time.time() - start_time) // 60)}min "
                 f"{(time.time() - start_time) % 60:.2f}s.")
    logging.info(f"Get datasets")

    ds_train = QueryDataset(
        # path_to_file="/users/iris/rserrano/data-set_pre_processed/fold-1/articles_train_all_ids.csv",
        path_to_file="../../../data-subset_pre_processed/fold-1/articles_train_all_ids.csv",
        text_column=text_column,
        query_encoder=dpr.encode_queries,
        corpus=corpus_train,
        nb_irrelevant=2
    )
    ds_val = QueryDataset(
        # path_to_file="/users/iris/rserrano/data-set_pre_processed/fold-2/articles_train_all_ids.csv",
        path_to_file="../../../data-subset_pre_processed/fold-2/articles_train_all_ids.csv",
        text_column=text_column,
        query_encoder=dpr.encode_queries,
        corpus=corpus_val
    )
    ds_test = QueryDataset(
        # path_to_file="/users/iris/rserrano/data-set_pre_processed/fold-3/articles_train_all_ids.csv",
        path_to_file="../../../data-subset_pre_processed/fold-3/articles_train_all_ids.csv",
        text_column=text_column,
        query_encoder=dpr.encode_queries,
        corpus=corpus_test
    )

    logging.info("elapsed time : "
                 f"{int((time.time() - start_time) // 60)}min "
                 f"{(time.time() - start_time) % 60:.2f}s.")
    logging.info(f"Send datasets to model")

    dpr.get_sets(train_val_test=(ds_train, ds_val, ds_test))

    logging.info("elapsed time : "
                 f"{int((time.time() - start_time) // 60)}min "
                 f"{(time.time() - start_time) % 60:.2f}s.")
    logging.info(f"Make logger and checkpoint's folders")

    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")

    logger = pl_loggers.TensorBoardLogger("./checkpoints", name="dpr_retriever")
    checkpoint_callback = ModelCheckpoint(monitor="Val/loss_epoch", mode="min", save_top_k=2, every_n_epochs=2)

    logging.info("elapsed time : "
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
                          log_every_n_steps=1)
    else:
        trainer = Trainer(logger=logger,
                          precision=32,
                          accelerator="cpu",
                          strategy='dp',
                          max_epochs=100,
                          callbacks=[checkpoint_callback],
                          log_every_n_steps=1)

    logging.info("elapsed time : "
                 f"{int((time.time() - start_time) // 60)}min "
                 f"{(time.time() - start_time) % 60:.2f}s.")
    logging.info(f"Fit trainer")
    trainer.fit(model=dpr)

    logging.info(f"DONE.".ljust(50), f";\telapsed time : "
                                     f"{int((time.time() - start_time) // 60)}min "
                                     f"{(time.time() - start_time) % 60:.2f}s.")


if __name__ == "__main__":
    main()
