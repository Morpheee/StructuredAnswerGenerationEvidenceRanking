#! /usr/bin/env python3
import torch
import os.path
from transformers import T5Tokenizer
from icecream import ic
import pandas as pd
from retriever.dpr_retriever import DPR
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import time


def df_get(path, text_column=None):
    df = pd.read_csv(path)
    if "corpus" in path :
        columns = ["id", "text"]
    else :
        columns = ["query", "outline", "text_"+text_column,
                   "paragraph_span_"+text_column,
                   "paragraph_id_" + text_column]
    df = df[columns]
    if not "corpus" in path :
        df = df.loc[df["outline"].map(len) > 0]
        df["text"] = df["text_"+text_column]
        df["paragraph_span"] = df["paragraph_span_"+text_column]
        df["paragraph_id"] = df["paragraph_id_"+text_column]
    df = df.loc[df["text"] != "\n"]
    return df


def main():
    start_time = time.time()

    text_column = "w/o_heading_first_sentence_by_paragraph"

    df_train = df_get(path="../../data_pre_processed/fold-1/articles_train.csv",
                      text_column=text_column)
    # corpus_train = pd.read_csv("../../data_pre_processed/fold-1/corpus_train.csv").sample(25)
    #
    # df_val = df_get("../../data_pre_processed/fold-2/articles_train.csv",
    #                 text_column=text_column)
    # corpus_val = pd.read_csv("../../data_pre_processed/fold-2/corpus_train.csv").sample(25)
    #
    # df_test = df_get("../../data_pre_processed/fold-3/articles_train.csv",
    #                  text_column=text_column)
    # corpus_test = pd.read_csv("../../data_pre_processed/fold-3/corpus_train.csv").sample(25)

    #---------------------------------------------------------------)

    dpr = DPR(context_model_name="facebook/dpr-ctx_encoder-single-nq-base",
              query_model_name="facebook/dpr-question_encoder-single-nq-base")

    dpr.encode_contexts(df_train)
    print(dpr.contexts.iloc[0])
    #
    # top_k_docs = []
    # for query in df_train["query"]:
    #     top_k_docs.append(dpr(query,
    #                           k=3,
    #                           return_contexts=True, ))
    # df_train["candidats"] = top_k_docs
    #
    # top_k_docs = []
    # for query in df_val["query"]:
    #     top_k_docs.append(dpr(query,
    #                           k=3,
    #                           return_contexts=True, ))
    # df_val["candidats"] = top_k_docs
    #
    # top_k_docs = []
    # for query in df_test["query"]:
    #     top_k_docs.append(dpr(query,
    #                           k=3,
    #                           return_contexts=True, ))
    # df_test["candidats"] = top_k_docs
    #
    # df_train.to_csv("train_temp.csv")
    # df_train.to_csv("val_temp.csv")
    # df_train.to_csv("test_temp.csv")
    # print(f"elapsed time - retriever : {time.time() - start_time:.2f}s. "
    #       f"i.e. {time.time() - start_time:.2f//60}min and {time.time() - start_time:.2f%60}s")
    #
    # start_time_generator = time.time()
    # tokenizer.add_tokens(['[Query:]', '[Documents:]', '[Document:]'])
    #
    # train_dataset = CustomDataset_planning.construct_from_raw(df_train, end2end=True)
    # print("TRAIN Dataset: {}".format(train_dataset.shape))
    # training_set = CustomDataset_planning(dataframe=train_dataset,
    #                                       tokenizer=tokenizer,
    #                                       source_len=512,
    #                                       summ_len=512,
    #                                       outline_len=128)
    # validation_set = CustomDataset_planning.construct_from_raw(df_train, end2end=True)
    # print("VAL Dataset: {}".format(validation_set.shape))
    # validation_set = CustomDataset_planning(dataframe=validation_set,
    #                                         tokenizer=tokenizer,
    #                                         source_len=512,
    #                                         summ_len=512,
    #                                         outline_len=128)
    # test_set = CustomDataset_planning.construct_from_raw(df_train, end2end=True)
    # print("TEST Dataset: {}".format(test_set.shape))
    # test_set = CustomDataset_planning(dataframe=test_set,
    #                                   tokenizer=tokenizer,
    #                                   source_len=512,
    #                                   summ_len=512,
    #                                   outline_len=128)
    #
    # gpus = -1
    #
    # nb_available_devices = 0  # = torch.cuda.device_count()
    # # accumulation_gradient = 1 // (1 * nb_available_devices)
    # accumulation_gradient = 1
    #
    # model = \
    #     T5Planning(train_val_test=(training_set, validation_set, test_set),
    #                train_batch_size=1,
    #                val_batch_size=1,
    #                lr=1e-4,
    #                cache_dir=os.path.join("./checkpoint", 'cache'),
    #                init_model=None,
    #                max_input=512, max_output=512, max_outline_len=128,
    #                model_name="t5-base", ptokenizer=tokenizer)
    #
    # tb_logger = pl_loggers.TensorBoardLogger("./checkpoint", name="project")
    # # checkpoint_callback = ModelCheckpoint(monitor="Val/loss_epoch", mode="min", save_top_k=2, every_n_epochs=2)
    #
    # s2s_trainer = Trainer(logger=tb_logger, precision=32, gpus=gpus,
    #                       accumulate_grad_batches=accumulation_gradient, max_epochs=100,
    #                       strategy='dp')  # , callbacks=[checkpoint_callback]) #try "dp" instead of "ddp"
    # s2s_trainer.fit(model)
    #
    # eval_trainer = Trainer(gpus=1)
    # predictions, references, outlines = model.predict(eval_trainer)
    # final_df = pd.DataFrame({'Generated Text': predictions, 'Generated Outlines': outlines, 'Actual Text': references})
    # final_df.to_csv("./outputs/output.csv")
    # print('Output Files generated for review')
    torch.cuda.empty_cache()
    # print(f"elapsed time - genrator : {time.time() - start_time_generator:.2f}s. "
    #       f"i.e. {(time.time() - start_time_generator) // 60:.2f}min " \
    #       f"and {(time.time() - start_time_generator) % 60:.2f}s")
    print(f"elapsed time - total : {time.time() - start_time:.2f}s. "
          f"i.e. {(time.time() - start_time) // 60:.2f}min and {(time.time() - start_time) % 60:.2f}s")


if __name__ == "__main__":
    main()
