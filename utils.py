import pickle
import os
import csv
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, MT5EncoderModel
from tqdm import tqdm


def save_data(data, name):
    with open('{}.pkl'.format(name), 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class Embedder:

    def __init__(self, device, transformers_models, embedding_dims,
                 use_model, layer_all, layer_rubert, model_dir):
        self.device = device
        self.transformers_models = transformers_models
        self.embedding_dims = embedding_dims
        self.use_model = use_model
        self.layer_all = layer_all
        self.layer_rubert = layer_rubert
        self.model_dir = model_dir

    @staticmethod
    def get_embedding_use(text, model):
        return model([text]).numpy()[0]

    def records_to_embeds_use(self, records, model, dim=512):
        embeddings = np.zeros((len(records), dim))
        for i, record in tqdm(enumerate(records), total=len(records)):
            embeddings[i] = self.get_embedding_use(record["title"], model)
        return embeddings

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_text(self, texts, tokenizer, model, max_tokens_count=100, layer=-1):
        encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=max_tokens_count,
                                  return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = model(encoded_input.input_ids)
        sentence_embeddings = self.mean_pooling(model_output.hidden_states[layer], encoded_input.attention_mask)
        return sentence_embeddings

    def get_embedding_transformers(self, text, tokenizer, model, layer):
        return self.embed_text([text], tokenizer, model, layer=layer)[0]

    def records_to_embeds_transformers(self, records, tokenizer, model, dim, layer=-1):
        embeddings = np.zeros((len(records), dim))
        for i, record in tqdm(enumerate(records), total=len(records)):
            embeddings[i] = self.get_embedding_transformers(record["title"], tokenizer, model, layer).cpu().numpy()
        return embeddings

    def records_to_embeds(self, model_name, records, dim, encoder_model=AutoModel, layer=-1):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        model = encoder_model.from_pretrained(model_name, config=config).to(self.device)

        return self.records_to_embeds_transformers(records, tokenizer, model, dim, layer)

    def get_all_embeddings(self, records, records_name):
        embeddings = {}

        print(f'Creating USE {records_name} embeddings')
        embeddings['use'] = self.records_to_embeds_use(records, self.use_model)

        for i, model_name in enumerate(self.transformers_models):
            print(f'Creating {model_name} {records_name} embeddings')
            if model_name == 'google/mt5-large':
                encoder_model = MT5EncoderModel
            else:
                encoder_model = AutoModel
            layer = self.layer_all
            if model_name == 'DeepPavlov/rubert-base-cased':
                layer = self.layer_rubert
            embeddings[model_name] = self.records_to_embeds(
                model_name, records, self.embedding_dims[i], encoder_model, layer)

        return embeddings

    def load_or_get(self, prefix, records, records_name):
        embeddings_path = os.path.join(self.model_dir, f'{prefix}_{self.layer_all}.pkl')
        if os.path.exists(embeddings_path):
            embeddings = load_data(embeddings_path)
        else:
            embeddings = self.get_all_embeddings(records, records_name)

        return embeddings


# Source: https://github.com/IlyaGusev/purano
def read_markup_tsv(file_name, clean_header=True):
    with open(file_name, "r") as r:
        header = tuple(next(r).strip().split("\t"))
        if clean_header:
            header = tuple((h.split(":")[-1] for h in header))
        reader = csv.reader(r, delimiter='\t', quotechar='"')
        records = [dict(zip(header, row)) for row in reader]
        clean_fields = ("first_title", "second_title", "first_text", "second_text")
        for record in records:
            for field in clean_fields:
                if field not in record:
                    continue
                record[field] = record.pop(field).strip().replace("\xa0", " ")
    return records
