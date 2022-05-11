#! /usr/bin/env python3
# https://github.com/huggingface/transformers/blob/main/examples/research_projects/rag-end2end-retriever/finetune_rag.py
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import pandas as pd
import torch
from torch import nn
from icecream import ic


class DPR(nn.Module):
    """
    Implementation of the DPR module :
    Encode all documents (contexts), and query with different BERT encoders.
    Similarity measure with dot product.
    """

    def __init__(self,
                 query_model_name: str,
                 context_model_name: str,
                 dense_size=64):
        super(DPR, self).__init__()
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(query_model_name)
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_model_name)
        self.query_model = DPRQuestionEncoder.from_pretrained(query_model_name)
        self.context_model = DPRContextEncoder.from_pretrained(context_model_name)
        self.contexts_encodings = None  # only used if exode_context -> store == True
        self.dense_contexts = None
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.contexts_to_dense = nn.Sequential(nn.Linear(768, dense_size * 2),
                                               nn.ReLU(),
                                               nn.Linear(dense_size * 2, dense_size),
                                               nn.GELU())

        self.query_to_dense = nn.Sequential(nn.Linear(768, dense_size * 2),
                                            nn.ReLU(),
                                            nn.Linear(dense_size * 2, dense_size),
                                            nn.GELU())

    def encode_contexts(self, contexts: list, store=True):
        contexts_encodings = self.context_tokenizer(contexts, truncation=True, padding=True, return_tensors='pt')
        if store:
            self.contexts_encodings = contexts_encodings
        return contexts_encodings

    def decode_contexts(self, contexts_encodings: list):
        contexts = [self.context_tokenizer.decode(c) for c in contexts_encodings]
        return contexts

    def get_dense_contexts(self, contexts=None, store=True):
        if contexts:
            contexts_encodings = self.encode_contexts(contexts)
        else:
            contexts_encodings = self.contexts_encodings
        dense_contexts = self.context_model(input_ids=contexts_encodings["input_ids"],
                                            attention_mask=contexts_encodings["attention_mask"])
        dense_contexts = self.contexts_to_dense(dense_contexts["pooler_output"])
        if store:
            self.dense_contexts = dense_contexts
        return dense_contexts

    def encode_query(self, query):
        query_encodings = self.query_tokenizer(query, truncation=True, padding=True, return_tensors='pt')
        return query_encodings

    def get_dense_query(self, query):
        query_encodings = self.encode_query(query)
        dense_query = self.query_model(input_ids=query_encodings["input_ids"],
                                       attention_mask=query_encodings["attention_mask"])
        dense_query = self.query_to_dense(dense_query["pooler_output"])
        return dense_query

    def dot_product(self, q_vector, p_vector):
        q_vector = q_vector.unsqueeze(1)
        sim = torch.matmul(q_vector, torch.transpose(p_vector, -2, -1))
        return sim

    def forward(self, query: str, contexts=None, return_contexts=False, k=0):
        dense_query = self.get_dense_query(query)

        if self.dense_contexts is not None:
            dense_contexts = self.dense_contexts
        else:
            dense_contexts = self.get_dense_contexts()

        similarity_score = self.dot_product(dense_query, dense_contexts)
        # logits = self.log_softmax(similarity_score).squeeze()
        logits = similarity_score.squeeze()
        if k == 0:
            top_k = logits.argsort(descending=True)
        else :
            top_k = logits.argsort()[:k]
        if return_contexts:
            if contexts:
                return contexts[top_k]
            else:
                top_k_docs = self.contexts_encodings["input_ids"][top_k]
                return self.decode_contexts(top_k_docs)
        else:
            return top_k

        # return logits, dense_query, dense_contexts
