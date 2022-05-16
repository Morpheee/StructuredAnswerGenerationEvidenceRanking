#! /usr/bin/env python3

import pandas as pd
from ..retriever.dpr_retriever import DPR
from ..generator.t5_generator import planningSeqToSeq
from transformers import (T5ForConditionalGeneration,
                          T5Tokenizer)
import os
from planning_model import Planning_Sequential
import warnings


class DPR_T5(planningSeqToSeq):
    """
    pipeline model, retriever and generator
    """

    def __init__(self,
                 retriever_params,
                 max_input=512, max_output=512, max_outline_len=128,
                 train_val_test=None,
                 train_batch_size=2, val_batch_size=2,
                 cache_dir=None, model_name='t5-base',
                 lr=1e-5, ptokenizer=None, init_model=None):
        super(DPR_T5, self).__init__()
        retriever_model = DPR(**retriever_params)

        try:
            eval_cache_dir = os.path.expandvars(cache_dir)
            text_model = T5ForConditionalGeneration.from_pretrained(
                os.path.join(eval_cache_dir, model_name + '-text-model'))
            if (init_model is not None):
                outline_model = T5ForConditionalGeneration.from_pretrained(init_model)
            else:
                outline_model = T5ForConditionalGeneration.from_pretrained(
                    os.path.join(eval_cache_dir, model_name + '-outline-model'))
            if ptokenizer is None:
                self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(eval_cache_dir, model_name + '-tokenizer'))
            else:
                self.tokenizer = ptokenizer

        except:
            print("No model found try to download it ")
            if (cache_dir is not None):
                eval_cache_dir = os.path.expandvars(cache_dir)
            else:
                eval_cache_dir = None
            text_model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=eval_cache_dir)
            if (init_model is not None):
                outline_model = T5ForConditionalGeneration.from_pretrained(init_model)
            else:
                outline_model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=eval_cache_dir)
            if ptokenizer is None:
                self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=eval_cache_dir)
            else:
                self.tokenizer = ptokenizer
            if (cache_dir is not None):
                text_model.save_pretrained(os.path.join(eval_cache_dir, model_name + '-text-model'))
                if (init_model is None):
                    outline_model.save_pretrained(os.path.join(eval_cache_dir, model_name + '-outline-model'))
                self.tokenizer.save_pretrained(os.path.join(eval_cache_dir, model_name + '-tokenizer'))

        self.model = Planning_Sequential(retriever_model=retriever_model,
                                         outline_model=outline_model,
                                         text_model=text_model)
        self.max_input, self.max_output, self.max_outline_len = max_input, max_output, max_outline_len

        if (train_val_test is not None):
            self.train_dt, self.val_dt, self.test_dt = (train_val_test[i] if (i < len(train_val_test)) else None
                                                        for i in range(3))
        else:
            warnings.warn("No dataset are given in the constructor, using multi-gpu will be unfeasible")
            self.train_dt, self.val_dt, self.test_dt = (None, None, None)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.lr = lr


    def training_step(self, data, batch_idx):
        self.model.train()
        y = data['target_ids']
        y_ids = y[:, :-1].contiguous()

        outline_y = data['outline_ids']
        outline_y_ids = outline_y[:, :-1].contiguous()
        outline_mask = data['outline_mask']

        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100

        outline_lm_labels = outline_y[:, 1:].clone().detach()
        outline_lm_labels[outline_y[:, 1:] == self.tokenizer.pad_token_id] = -100

        ids = data['source_ids']
        mask = data['source_mask']

        output = self.model(input_ids={'source_ids': ids, 'outline_ids': outline_y},
                            attention_mask={'source_mask': mask, 'outline_mask': outline_mask},
                            decoder_input_ids={'outline_ids': outline_y_ids, 'ful_answer_ids': y_ids},
                            labels={'outline_labels': outline_lm_labels, 'full_answer_labels': lm_labels})
        loss = output['sum_loss']

        self.log('train/loss_step', loss.item(), on_step=True)
        self.log('train/loss_epoch', loss.item(), on_step=False, on_epoch=True)

        return loss

    def training_epoch_end(self, outputs):
        print('Finishing epoch ', self.current_epoch)

    def validation_step(self, data, batch_idx):
        self.model.eval()
        with torch.no_grad():
            y = data['target_ids']
            y_ids = y[:, :-1].contiguous()

            outline_y = data['outline_ids']
            outline_mask = data['outline_mask']
            outline_y_ids = outline_y[:, :-1].contiguous()

            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100

            outline_lm_labels = outline_y[:, 1:].clone().detach()
            outline_lm_labels[outline_y[:, 1:] == self.tokenizer.pad_token_id] = -100

            ids = data['source_ids']
            mask = data['source_mask']

            output = self.model(input_ids={'source_ids': ids, 'outline_ids': outline_y},
                                attention_mask={'source_mask': mask, 'outline_mask': outline_mask},
                                decoder_input_ids={'outline_ids': outline_y_ids,
                                                   'ful_answer_ids': y_ids},
                                labels={'outline_labels': outline_lm_labels, 'full_answer_labels': lm_labels},
                                validation=True)
            loss = output['sum_loss']

            self.log('Val/loss_step', loss.item(), on_step=True)
            self.log('Val/loss_epoch', loss.item(), on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs):
        print('Validation epoch ', self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, data, batch_idx):
        self.model.eval()
        with torch.no_grad():
            y = data['target_ids']

            outline_y = data['outline_ids']

            outline_lm_labels = outline_y[:, 1:].clone().detach()
            outline_lm_labels[outline_y[:, 1:] == self.tokenizer.pad_token_id] = -100

            ids = data['source_ids']
            mask = data['source_mask']

            gen = self.model.generate(input_ids=ids,
                                      attention_mask=mask,
                                      max_output_size=512,
                                      MAX_OUTLINE_LEN=self.max_outline_len,
                                      num_beams=4,
                                      repetition_penalty=2.5,
                                      length_penalty=1.0,
                                      early_stopping=True)
            preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     gen['full_answer_ids']]
            outl = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                    gen['outline_ids']]
            target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if batch_idx % 100 == 0:
                print("Completed", batch_idx)
        return preds, target, outl
