import torch
import sys
import random
import copy
from catboost import Pool, CatBoost
import tensorflow_text
import tensorflow_hub as hub

from utils import *


def get_groups(markup):
    groups = list()
    url2group = dict()
    for r in markup:
        if r["quality"] == "draw" or r["quality"] == "bad":
            continue
        left_url = r["left_url"]
        right_url = r["right_url"]
        left_group_id = url2group.get(left_url, None)
        right_group_id = url2group.get(right_url, None)
        if left_group_id is not None and right_group_id is None:
            groups[left_group_id].add(right_url)
            url2group[right_url] = left_group_id
        elif right_group_id is not None and left_group_id is None:
            groups[right_group_id].add(left_url)
            url2group[left_url] = right_group_id
        elif left_group_id is None and right_group_id is None:
            groups.append({left_url, right_url})
            url2group[left_url] = url2group[right_url] = len(groups) - 1
        elif left_group_id != right_group_id:
            for u in groups[right_group_id]:
                url2group[u] = left_group_id
            groups[left_group_id] = groups[left_group_id] | groups[right_group_id]
            groups[right_group_id] = set()
        assert right_url in groups[url2group[right_url]]
        assert left_url in groups[url2group[left_url]]
        assert url2group[right_url] == url2group[left_url]
    groups = [group for group in groups if group]
    url2group = dict()
    for i, group in enumerate(groups):
        for url in group:
            url2group[url] = i
    return groups, url2group


def get_features(markup, embeddings, url2id):
    urls = set()
    for r in markup:
        if r["quality"] == "draw" or r["quality"] == "bad":
            continue
        urls.add(r["left_url"])
        urls.add(r["right_url"])
    features = dict()
    for url in urls:
        embedding = list(embeddings[url2id[url]])
        features[url] = embedding
    return features


def get_pairs(markup, url2group, is_train_group):
    all_pairs = list()
    train_pairs = list()
    val_pairs = list()
    for r in markup:
        if r["quality"] == "draw" or r["quality"] == "bad":
            continue
        left_url = r["left_url"]
        right_url = r["right_url"]
        group_id = url2group[left_url]
        assert url2group[right_url] == group_id
        is_train = is_train_group[group_id]
        pairs = train_pairs if is_train else val_pairs
        if r["quality"] == "left":
            pairs.append((left_url, right_url))
            all_pairs.append((left_url, right_url))
        elif r["quality"] == "right":
            pairs.append((right_url, left_url))
            all_pairs.append((right_url, left_url))
    return all_pairs, train_pairs, val_pairs


def convert_to_cb_pool(groups, pairs, features):
    urls_set = set()
    for url1, url2 in pairs:
        urls_set.add(url1)
        urls_set.add(url2)
    urls = []
    group_id = []
    for i, group in enumerate(groups):
        for url in group:
            if url not in urls_set:
                continue
            group_id.append(i)
            urls.append(url)
    urls2id = {url: i for i, url in enumerate(urls)}
    features = [features[url] for url in urls]
    pairs = [(urls2id[url1], urls2id[url2]) for url1, url2 in pairs]
    pool = Pool(data=features, pairs=pairs, group_id=group_id)
    return pool, urls


def train_classifier(train_title_embeddings, hl_train_markup, train_url2id, url2group, is_train_group, groups):
    features = get_features(hl_train_markup, train_title_embeddings, train_url2id)
    all_pairs, train_pairs, val_pairs = get_pairs(hl_train_markup, url2group, is_train_group)

    train_pool, train_urls = convert_to_cb_pool(groups, train_pairs, features)
    val_pool, val_urls = convert_to_cb_pool(groups, val_pairs, features)

    cb_model = CatBoost(
        params={"loss_function": "PairLogit", "task_type": "GPU", "iterations": 1000, "logging_level": "Silent"})
    cb_model.fit(train_pool, eval_set=val_pool)
    return cb_model, val_pool, val_urls, val_pairs


def accuracy(cb_model, val_pool, val_urls, val_pairs, draw_border=0.1):
    val_predictions = cb_model.predict(val_pool)
    val_urls2id = {url: i for i, url in enumerate(val_urls)}
    correct_pairs = 0
    all_pairs = 0
    for winner_url, loser_url in val_pairs:
        all_pairs += 1
        winner_id = val_urls2id[winner_url]
        loser_id = val_urls2id[loser_url]
        winner_pred = val_predictions[winner_id]
        loser_pred = val_predictions[loser_id]

        if abs(winner_pred - loser_pred) < draw_border:
            correct_pairs += 0.5
        elif winner_pred > loser_pred:
            correct_pairs += 1

    return float(correct_pairs) * 100 / all_pairs


def blend_accuracy(model_names, cb_models, val_pools, val_urls, val_pairs, draw_border=0.1):
    val_predictions = {}
    for model_name in model_names:
        val_predictions[model_name] = cb_models[model_name].predict(val_pools[model_name])

    val_urls2id = {url: i for i, url in enumerate(val_urls)}
    correct_pairs = 0
    all_pairs = 0
    for winner_url, loser_url in val_pairs:
        all_pairs += 1
        winner_id = val_urls2id[winner_url]
        loser_id = val_urls2id[loser_url]

        winner_preds = {}
        loser_preds = {}
        for model_name in model_names:
            winner_preds[model_name] = val_predictions[model_name][winner_id]
            loser_preds[model_name] = val_predictions[model_name][loser_id]

        winner_pred = np.mean(list(winner_preds.values()))
        loser_pred = np.mean(list(loser_preds.values()))

        if abs(winner_pred - loser_pred) < draw_border:
            correct_pairs += 0.5
        elif winner_pred > loser_pred:
            correct_pairs += 1

    return float(correct_pairs) * 100 / all_pairs


def test_pred(markup, records, test_embeddings, cb_models, dataset_name, draw_border=0.1):
    predictions = {}
    for model_name in test_embeddings.keys():
        predictions[model_name] = cb_models[model_name].predict(test_embeddings[model_name])

    url2idx = dict()
    for i, record in enumerate(records):
        url = record["url"]
        url2idx[url] = i

    submission = []
    for record in markup:
        if record["dataset"] != dataset_name:
            continue

        left_url = record["left_url"]
        right_url = record["right_url"]
        left_idx = url2idx[left_url]
        right_idx = url2idx[right_url]

        left_preds = {}
        right_preds = {}
        for model_name in predictions.keys():
            left_preds[model_name] = predictions[model_name][left_idx]
            right_preds[model_name] = predictions[model_name][right_idx]

        left_pred = np.mean(list(left_preds.values()))
        right_pred = np.mean(list(right_preds.values()))

        result = "left" if left_pred > right_pred else "right"
        result = "draw" if abs(left_pred - right_pred) < draw_border else result

        r = copy.copy(record)
        r["quality"] = result
        submission.append(r)

    return submission


def evaluate(preds, markup):
    correct = 0
    count = 0
    for pred, gold in zip(preds, markup):
        assert pred['left_url'] == gold['left_url'] and pred['right_url'] == gold['right_url']
        if gold['quality'] == 'bad':
            continue
        count += 1
        if pred['quality'] == gold['quality']:
            correct += 1
        elif gold['quality'] == 'draw' or pred['quality'] == 'draw':
            correct += 0.5

    return correct * 100 / count


def evaluate_single(test_embeddings, hl_test_markup, test_records, gold_markup, test_name, model_names, cb_models):
    for model_name in model_names:
        single_embs = {model_name: test_embeddings[model_name]}
        single_cb_model = {model_name: cb_models[model_name]}
        preds = test_pred(hl_test_markup, test_records, single_embs, single_cb_model, test_name)
        print(f'test {test_name} {model_name} accuracy: {evaluate(preds, gold_markup):.2f}')


def main():
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = '.'

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    random.seed(42)

    train_records = load_data(os.path.join(model_dir, 'train_records.pkl'))
    test_0527_records = load_data(os.path.join(model_dir, 'test_0527_records.pkl'))
    test_0529_records = load_data(os.path.join(model_dir, 'test_0529_records.pkl'))
    hl_train_markup = read_markup_tsv("titles_markup_0525_urls.tsv")
    hl_test_markup = read_markup_tsv("titles_markup_test.tsv")

    transformers_models = ['xlm-roberta-large', 'sberbank-ai/sbert_large_nlu_ru',
                           'google/mt5-large', 'DeepPavlov/rubert-base-cased']
    model_names = transformers_models + ['use']
    dims = [1024, 1024, 1024, 768]

    layer_all = 19
    layer_rubert = 8

    train_url2id = {r["url"]: i for i, r in enumerate(train_records)}
    val_part = 0.1
    train_part = 1.0 - val_part
    groups, url2group = get_groups(hl_train_markup)
    is_train_group = [random.random() < train_part for _ in range(len(groups))]

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    use_model = hub.load(module_url)
    print("module %s loaded" % module_url)

    embedder = Embedder(device, transformers_models, dims, use_model, layer_all, layer_rubert, model_dir)
    embeddings_train = embedder.load_or_get('embeddings_train', train_records, 'train')

    cb_models = {}
    val_pools = {}
    val_urls = {}
    val_pairs = {}
    for model_name in model_names:
        print(f'Training classifier on {model_name}')
        cb_model, val_pool, val_url, val_pair = train_classifier(
            embeddings_train[model_name], hl_train_markup, train_url2id, url2group, is_train_group, groups)
        cb_models[model_name] = cb_model
        val_pools[model_name] = val_pool
        val_urls[model_name] = val_url
        val_pairs[model_name] = val_pair

    for model_name in model_names:
        print(f'Val accuracy for {model_name}: '
              f'{accuracy(cb_models[model_name], val_pools[model_name], val_urls[model_name], val_pairs[model_name]):.2f}')
    print(f'Blend val accuracy: {blend_accuracy(model_names, cb_models, val_pools, val_urls["use"], val_pairs["use"]):.2f}')

    test_0527_embeddings = embedder.load_or_get('test_0527_embeddings', test_0527_records, 'test_0527')
    test_0529_embeddings = embedder.load_or_get('test_0529_embeddings', test_0529_records, 'test_0529')

    gold_markup_0527 = read_markup_tsv(os.path.join(model_dir, "titles_markup_0527_urls.tsv.txt"))
    gold_markup_0529 = read_markup_tsv(os.path.join(model_dir, "titles_markup_0529_urls.tsv.txt"))

    evaluate_single(test_0527_embeddings, hl_test_markup, test_0527_records, gold_markup_0527, '0527',
                    model_names, cb_models)
    evaluate_single(test_0529_embeddings, hl_test_markup, test_0529_records, gold_markup_0529, '0529',
                    model_names, cb_models)

    preds_0527 = test_pred(hl_test_markup, test_0527_records, test_0527_embeddings, cb_models, '0527')
    print(f'test 0527 blend accuracy: {evaluate(preds_0527, gold_markup_0527):.2f}')
    preds_0529 = test_pred(hl_test_markup, test_0529_records, test_0529_embeddings, cb_models, '0529')
    print(f'test 0529 blend accuracy: {evaluate(preds_0529, gold_markup_0529):.2f}')


if __name__ == "__main__":
    main()
