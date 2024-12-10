import pandas as pd
import re
import json
from typing import Dict, List
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertModel, BertTokenizer
import torch


def expand_abbreviations(text: str, abbrev_dict: Dict[str, str]) -> str:
    for abbrev, expanded in abbrev_dict.items():
        text = text.replace(abbrev, expanded)
    return text

def lower_case(text: str) -> str:
    return text.lower()


def remove_urls(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    return re.sub(r'\w+.com\s?', '', text)


def replace_contraction(text: str, contraction_dict: Dict[str, str]) -> str:
    for contraction, expanded in contraction_dict.items():
        text = text.replace(contraction, expanded)
    return text


def remove_stopwords(text: str, stopwords: List[str]) -> str:
    words = text.split()
    return " ".join([word for word in words if word not in stopwords])


def remove_special_characters(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))


def remove_whitespace(text: str) -> str:
    return " ".join(text.split())


def remove_tags(text: str) -> str:
    return re.sub(r'@\S+', '', text)


def preprocess(
    text: str, 
    stopwords: List[str], 
    abbreviation_dict: Dict[str, str], 
    contraction_dict: Dict[str, str]
) -> str:
    text = expand_abbreviations(text, abbreviation_dict)
    text = remove_tags(text)
    text = lower_case(text)
    text = remove_urls(text)
    text = replace_contraction(text, contraction_dict)
    text = remove_stopwords(text, stopwords)
    text = remove_special_characters(text)
    text = remove_whitespace(text)
    return text


def bert_embedding(
    text: str, 
    tokenizer: BertTokenizer, 
    model: BertModel
) -> List[float]:
    tokens = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    T = 120
    padded_tokens = tokens + ['[PAD]' for _ in range(T - len(tokens))]
    attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
    seg_ids = [0 for _ in range(len(padded_tokens))]
    sent_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
    token_ids = torch.tensor(sent_ids).unsqueeze(0)
    attn_mask = torch.tensor(attn_mask).unsqueeze(0)
    seg_ids = torch.tensor(seg_ids).unsqueeze(0)
    output = model(token_ids, attention_mask=attn_mask, token_type_ids=seg_ids)
    last_hidden_state = output[1].detach().numpy()[0]
    return last_hidden_state.tolist()


def main():
    N = 15000
    df = pd.read_csv("data.csv", encoding="latin_1", names=['target', 'ids', 'date', 'flag', 'user', 'text'])[['text', 'target']]
    posetive_df = df[df['target'] == 4].sample(n=int(N / 2), random_state=0)
    negative_df = df[df['target'] == 0].sample(n=int(N / 2), random_state=0)
    df = pd.concat([posetive_df, negative_df]).sample(frac=1, random_state=0, ignore_index=True)
    df['target'] = df['target'].apply(lambda x: int(x / 2 - 1))

    with open('stopwords.json') as f:
        stopwords = json.load(f)
    with open('abbreviation.json') as f:
        abbreviation_dict = json.load(f)
    with open('contraction.json') as f:
        contraction_dict = json.load(f)

    vectorizer = CountVectorizer(max_features=512)
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    df['clean_text'] = df['text'].apply(lambda x: preprocess(x, stopwords, abbreviation_dict, contraction_dict)).dropna()
    df['bow'] = vectorizer.fit_transform(df['clean_text']).toarray().tolist()

    df['bert'] = df['clean_text'].apply(lambda x: bert_embedding(str(x), tokenizer, model))
    df.to_csv(f'embeddings_15000.csv')


if __name__ == "__main__":
    main()
