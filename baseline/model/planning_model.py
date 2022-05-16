import torch
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.models.t5.modeling_t5 import T5PreTrainedModel

from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'


class Planning_Sequential(nn.Module):
    def __init__(self, retriever_model, outline_model, text_model):
        super().__init__()
        self.retriever_model = retriever_model
        self.outline_model = outline_model
        self.text_model = text_model

    def save_pretrained(self, path):
        self.retriever_model.save_pretrained(path + "-retriever_model.ckpt")
        self.outline_model.save_pretrained(path + "-outline_model.ckpt")
        self.text_model.save_pretrained(path + "-text_model.ckpt")

    def forward(
            self,
            input_ids=None,
            decoder_input_ids=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            validation=False,
            return_dict=None,
            HR_batch=None):

        concat_inputs = torch.cat((input_ids['outline_ids'], input_ids['source_ids']), 1)
        text_outputs = self.text_model(input_ids=concat_inputs, decoder_input_ids=decoder_input_ids['ful_answer_ids'],
                                       labels=labels['full_answer_labels'])
        text_loss = text_outputs[0]
        return {
            'text_outputs': text_outputs,
            'sum_loss': text_loss,
            'text_loss': text_outputs[0],
        }

    def generate(self, input_ids=None,
                 attention_mask=None,
                 max_output_size=512,
                 MAX_OUTLINE_LEN=100,
                 num_beams=4,
                 repetition_penalty=2.5,
                 length_penalty=1.0,
                 early_stopping=True,
                 HR_batch=None):
        if (HR_batch is not None):
            input_ids_o, input_mask_o, decoder_input_ids_o, label_o = \
                self.outline_model.batch_transform(HR_batch)
            input_dict = {'input_ids': input_ids_o.to(device, dtype=torch.long),
                          'attention_mask': input_mask_o.to(device, dtype=torch.long),
                          'max_length': MAX_OUTLINE_LEN,
                          'num_beams': num_beams,
                          'repetition_penalty': repetition_penalty,
                          'length_penalty': length_penalty,
                          'early_stopping': early_stopping}
            generated_ids_outline = self.outline_model(input_dict)
        else:
            generated_ids_outline = self.outline_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_OUTLINE_LEN,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                early_stopping=True,
            )

        concat_ids = torch.cat((generated_ids_outline, input_ids), 1)
        generated_ids = self.text_model.generate(
            input_ids=concat_ids,
            max_length=max_output_size,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping
        )
        return {
            "outline_ids": generated_ids_outline,
            "full_answer_ids": generated_ids
        }
