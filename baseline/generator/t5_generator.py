#! /usr/bin/env python3

import os
import torch
import copy
import pandas as pd
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import warnings
from icecream import ic
sys.path.append("/home/legmint/Documents/MLDM/S4/internship/Complex-Answer-Generation")
from cogecsea.modules.planning_model import Planning_Sequential


class SeqToSeq(pl.LightningModule):
    '''Sequence to Sequence Model using pytorch lighning (Template).

    '''

    def __init__(self):
        super().__init__()

    def add_tokens(self, token):
        self.tokenizer.add_tokens([token])

    @staticmethod
    def collate_fn(list_output):
        queries, documents, labels = [], [], []

        for q, d, l in list_output:
            queries.append(q)
            documents.append(d)
            labels.append(l)
        return queries, documents, labels

    def train_dataloader(self):
        return \
            DataLoader(copy.deepcopy(self.train_dt), batch_size=self.train_batch_size,
                       drop_last=True, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return \
            DataLoader(copy.deepcopy(self.val_dt), batch_size=self.val_batch_size,
                       num_workers=4)

    def test_dataloader(self):
        return \
            DataLoader(copy.deepcopy(self.test_dt), batch_size=self.val_batch_size,
                       num_workers=4)

    def forward(self, input_dict):
        prediction = self.model.generate(**input_dict)
        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_epoch_end(self, outputs):
        self.test_predictions = sum([output[0] for output in outputs], [])
        self.test_actual = sum([output[1] for output in outputs], [])

    def predict(self, trainer):
        trainer.test(self)
        return self.test_predictions, self.test_actual


class planningSeqToSeq(pl.LightningModule):
    '''Planning Sequence to Sequence Model using pytorch lighning (Template).

    '''

    def __init__(self):
        super().__init__()

    def add_tokens(self, token):
        self.tokenizer.add_tokens([token])

    @staticmethod
    def collate_fn(list_output):
        queries, documents, labels = [], [], []

        for q, d, l in list_output:
            queries.append(q)
            documents.append(d)
            labels.append(l)
        return queries, documents, labels

    def train_dataloader(self):
        return \
            DataLoader(copy.deepcopy(self.train_dt), batch_size=self.train_batch_size,
                       drop_last=True, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return \
            DataLoader(copy.deepcopy(self.val_dt), batch_size=self.val_batch_size,
                       num_workers=4)

    def test_dataloader(self):
        return \
            DataLoader(copy.deepcopy(self.test_dt), batch_size=self.val_batch_size,
                       num_workers=4)

    def forward(self, input_dict):
        prediction = self.model.generate(**input_dict)
        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_epoch_end(self, outputs):
        self.test_predictions = sum([output[0] for output in outputs], [])
        self.test_actuals = sum([output[1] for output in outputs], [])
        self.test_outlines = sum([output[2] for output in outputs], [])

    def predict(self, trainer):
        trainer.test(self)
        return self.test_predictions, self.test_actuals, self.test_outlines


class T5SeqToSeq(SeqToSeq):
    '''T5 implementation of SeqToSeq.

    '''

    def __init__(self, max_input=512, max_output=512,
                 train_val_test=None,
                 train_batch_size=2, val_batch_size=2,
                 cache_dir=None, model_name='t5-base',
                 lr=1e-5, ptokenizer=None):
        '''
            Parameters
            ----------
            max_input : int, optional(default=512)
                Maximum input size to consider, notice that the model has
                16384 positional embeddings (please consider max_input < 16384)
                (default=1024)

            max_output : int, optional(default=256)
                Maximum output size to consider, notice that the model has
                16384 positional embeddings, moreover the attention for output is
                not memory efficient (default=256)

            train_val_test : tuple(Dataset, Dataset, Dataset), optional(default: None)
                the train validation and test set, if not given consider
                to specify them in the Trainer (however multi-gpu is unfeasible
                given dataset in the Trainer)

            train_batch_size : int, optional(default: 2)
                The batch size for training step

            val_batch_size : int, optional(default: 4)
                The default batch size for validation and testing set

            lr : float, optional(default=1e-5)
                Learning rate for low-level encoder

            model_name : str, optional(default='t5-small')
                Name of the T5 model configuration to use
        '''
        super().__init__()
        try:
            eval_cache_dir = os.path.expandvars(cache_dir)
            self.model = T5ForConditionalGeneration.from_pretrained(os.path.join(eval_cache_dir, model_name + '-model'))
            self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(eval_cache_dir, model_name + '-tokenizer'))
        except:
            print("No model found try to download it ")
            if (cache_dir is not None):
                eval_cache_dir = os.path.expandvars(cache_dir)
            else:
                eval_cache_dir = None
            self.model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=eval_cache_dir)
            if ptokenizer is None:
                self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=eval_cache_dir)
            else:
                self.tokenizer = ptokenizer
            if (cache_dir is not None):
                self.model.save_pretrained(os.path.join(eval_cache_dir, model_name + '-model'))
                self.tokenizer.save_pretrained(os.path.join(eval_cache_dir, model_name + '-tokenizer'))

        self.max_input, self.max_output = max_input, max_output

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
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
        ids = data['source_ids']
        mask = data['source_mask']

        # outputs = self.model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
        outputs = self.model(input_ids=ids, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]

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
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
            ids = data['source_ids']
            mask = data['source_mask']

            # outputs = self.model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            outputs = self.model(input_ids=ids, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]

            self.log('Val/loss_step', loss.item(), on_step=True)
            self.log('Val/loss_epoch', loss.item(), on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs):
        print('Validation epoch ', self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, data, batch_idx):
        self.model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            y = data['target_ids']
            ids = data['source_ids']
            mask = data['source_mask']

            generated_ids = self.model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=self.max_output,
                num_beams=4,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            prediction = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                          generated_ids]
            actual = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if batch_idx % 100 == 0:
                print("Completed", batch_idx)
        return prediction, actual


class T5Planning(planningSeqToSeq):
    '''T5 implementation of planningSeqToSeq.

    '''

    def __init__(self, max_input=512, max_output=512, max_outline_len=128,
                 train_val_test=None,
                 train_batch_size=2, val_batch_size=2,
                 cache_dir=None, model_name='t5-base',
                 lr=1e-5, ptokenizer=None, init_model=None):
        '''
            Parameters
            ----------
            max_input : int, optional(default=1024)
                Maximum input size to consider, notice that the model has
                16384 positional embeddings (please consider max_input < 16384)
                (default=1024)

            max_output : int, optional(default=256)
                Maximum output size to consider, notice that the model has
                16384 positional embeddings, moreover the attention for output is
                not memory efficient (default=256)

            train_val_test : tuple(Dataset, Dataset, Dataset), optional(default: None)
                the train validation and test set, if not given consider
                to specify them in the Trainer (however multi-gpu is unfeasible
                given dataset in the Trainer)

            train_batch_size : int, optional(default: 2)
                The batch size for training step

            val_batch_size : int, optional(default: 4)
                The default batch size for validation and testing set

            lr : float, optional(default=1e-5)
                Learning rate for low-level encoder

            model_name : str, optional(default='t5-small')
                Name of the T5 model configuration to use

            init_model: file path (default: None)
            path to a pretrained outline-generation model.
        '''
        super().__init__()

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
        self.model = Planning_Sequential(outline_model=outline_model, text_model=text_model)
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
                            # attention_mask={'source_mask': mask, 'outline_mask': outline_mask},
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
                                # attention_mask={'source_mask': mask, 'outline_mask': outline_mask},
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


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.output
        self.ctext = self.data.input

    @classmethod
    def construct_from_raw(cls, df, only_outline=False, language_model=False, end2end=False):

        def parse_outline(outline_list):
            out = []
            for o in outline_list:
                titles = o.split("/")
                for i in range(len(titles)):
                    balise_start = "[h" + str(i + 1) + "]"
                    balise_end = "[h" + str(i + 1) + "]"
                    if (not (balise_start + titles[i] + balise_end) in out):
                        out.append(balise_start + titles[i] + balise_end)
            return " ".join(out)

        inputs = []
        outputs = []
        outlines = []
        filtered = df.loc[df['outline'].str.len() != 0]
        for i_, row in filtered.iterrows():
            if (language_model):
                inputs.append("[Query:]" + row['query'])
            else:
                inputs.append(
                    "[Query:]" + row['query'] + "[Documents:]" + "".join(["[Document:]" + c for c in row['candidats']]))
            if (only_outline):
                outputs.append(parse_outline(row['outline']))
            else:
                outputs.append(row['text'])
                outlines.append(parse_outline(row['outline']))
            if (end2end):
                d = {'input': inputs, 'output': outputs, 'outline': outlines}
            else:
                d = {'input': inputs, 'output': outputs}
        df = pd.DataFrame(data=d)
        return df

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], truncation=True, max_length=self.source_len,
                                                  padding='max_length', return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], truncation=True, max_length=self.summ_len,
                                                  padding='max_length', return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class CustomDataset_planning(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len, outline_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.outline_len = outline_len
        self.text = self.data.output
        self.ctext = self.data.input
        self.outline = self.data.outline

    @classmethod
    def construct_from_raw(cls, df, only_outline=False, language_model=False, end2end=False):

        def parse_outline(outline_list):
            out = []
            for o in outline_list:
                titles = o.split("/")
                for i in range(len(titles)):
                    balise_start = "[h" + str(i + 1) + "]"
                    balise_end = "[h" + str(i + 1) + "]"
                    if (not (balise_start + titles[i] + balise_end) in out):
                        out.append(balise_start + titles[i] + balise_end)
            return " ".join(out)

        inputs = []
        outputs = []
        outlines = []
        filtered = df.loc[df['outline'].str.len() != 0]
        for i_, row in filtered.iterrows():
            if (language_model):
                inputs.append("[Query:]" + row['query'])
            else:
                inputs.append(
                    "[Query:]" + row['query'] + "[Documents:]" + "".join(["[Document:]" + c for c in row['candidats']]))
            if (only_outline):
                outputs.append(parse_outline(row['outline']))
            else:
                outputs.append(row['text'])
                outlines.append(parse_outline(row['outline']))
            if (end2end):
                d = {'input': inputs, 'output': outputs, 'outline': outlines}
            else:
                d = {'input': inputs, 'output': outputs}
        df = pd.DataFrame(data=d)
        return df

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        outline = str(self.outline[index])
        outline = ' '.join(outline.split())

        source = self.tokenizer.batch_encode_plus([ctext], truncation=True, max_length=self.source_len,
                                                  padding='max_length', return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], truncation=True, max_length=self.summ_len,
                                                  padding='max_length', return_tensors='pt')
        outline = self.tokenizer.batch_encode_plus([outline], truncation=True, max_length=self.outline_len,
                                                   padding='max_length', return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        outline_ids = outline['input_ids'].squeeze()
        outline_mask = outline['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'outline_ids': outline_ids.to(dtype=torch.long),
            'outline_mask': outline_mask.to(dtype=torch.long)
        }
