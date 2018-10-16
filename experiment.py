import sys
import collections
import random
import numpy
import os
import math
import operator
import scipy
import gc

import config_parser
from model import WordPairClassifier

unk_token = "<unk>"


def read_dataset(dataset_path):
    dataset = []
    for path in dataset_path.split(","):
        with open(path, 'r') as f:
            for line in f:
                line_parts = line.strip().split("\t")
                #assert(len(line_parts) == 3)
                dataset.append((float(line_parts[0]), line_parts[1], line_parts[2], [float(x) for x in line_parts[3:]]))
    return dataset


def construct_vocabulary(datasets, embedding_vocab_set):
    vocab = [unk_token]
    vocab_set = set()
    for dataset in datasets:
        for entry in dataset:
            for word in [entry[1], entry[2]]:
                if word not in vocab_set and (embedding_vocab_set == None or word in embedding_vocab_set):
                    vocab_set.add(word)
                    vocab.append(word)
    return vocab


def construct_embedding_vocab(datasets):
    embedding_vocab_set = set()
    for dataset in datasets:
        if dataset == None or len(dataset) == 0:
            continue
        with open(dataset, 'r') as f:
            for line in f:
                line_parts = line.strip().split()
                if len(line_parts) <= 2:
                    continue
                word = line_parts[0]
                if word not in embedding_vocab_set:
                    embedding_vocab_set.add(word)
    return embedding_vocab_set


def load_embeddings_into_matrix(embedding_path, shared_matrix, word2id):
    embedding_matrix = shared_matrix.get_value()
    vector_length = embedding_matrix.shape[1]
    with open(embedding_path, 'r') as f:
        line_length = None
        for line in f:
            line_parts = line.strip().split()
            if len(line_parts) <= 2:
                continue
            assert(len(line_parts) == vector_length+1), str(len(line_parts)) + " \t" + str(vector_length) + "\t" + line
            if line_parts[0] in word2id:
                embedding_matrix[word2id[line_parts[0]]] = numpy.array([float(x) for x in line_parts[1:]])
    shared_matrix.set_value(embedding_matrix)


def extend_vocabulary(vocabulary, path1, path2, main_separator, remove_multiword):
    vocab_set = set(vocabulary)
    for path in [path1, path2]:
        if path != None and len(path) > 0:
            with open(path, 'r') as f:
                for line in f:
                    line_parts = line.strip().split(main_separator, 1)
                    if len(line_parts) < 2:
                        continue
                    if remove_multiword == True and len(line_parts[0].split()) > 1:
                        continue
                    if line_parts[0] not in vocab_set:
                        vocab_set.add(line_parts[0])
                        vocabulary.append(line_parts[0])


def evaluate(cost, all_predicted_scores, all_gold_labels, name):
    assert(len(all_predicted_scores) == len(all_gold_labels))

    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0

    for i in range(len(all_predicted_scores)):
        if all_gold_labels[i] >= 5.0:
            if all_predicted_scores[i] >= 5.0:
                tp += 1.0
            else:
                fn += 1.0
        elif all_gold_labels[i] < 5.0:
            if all_predicted_scores[i] >= 5.0:
                fp += 1.0
            else:
                tn += 1.0

    assert(int(tp + fn + fp + tn) == len(all_predicted_scores))

    results = collections.OrderedDict()
    results[name + "_cost"] = cost
    results[name + "_count"] = len(all_predicted_scores)
    results[name + "_spearmanr"] = scipy.stats.spearmanr(all_gold_labels, all_predicted_scores)[0]
    results[name + "_pearsonr"] = scipy.stats.pearsonr(all_gold_labels, all_predicted_scores)[0]

    results[name + "_count_correct"] = tp + tn
    results[name + "_count_total"] = len(all_predicted_scores)
    results[name + "_tp"] = tp
    results[name + "_tn"] = tn
    results[name + "_fp"] = fp
    results[name + "_fn"] = fn
    results[name + "_accuracy"] = float(tp + tn) / float(len(all_predicted_scores))
    p = (tp / (tp + fp)) if (tp + fp) > 0.0 else 0.0
    r = (tp / (tp + fn)) if (tp + fn) > 0.0 else 0.0
    results[name + "_p"] = p
    results[name + "_r"] = r
    results[name + "_fmeasure"] = (2.0 * p * r / (p+r)) if (p+r) > 0.0 else 0.0
    return results


def process_dataset(dataset, model, word2id, is_testing, is_pretraining, config, name):
    if is_testing == False and config["shuffle_training_data"] == True:
        random.shuffle(dataset)

    cost_sum = 0.0
    all_predicted_scores, all_gold_labels = [], []
    for i in range(0, len(dataset), config["examples_per_batch"]):
        batch = dataset[i:i+config["examples_per_batch"]]
        if is_testing == False and config["shuffle_training_data"] == True:
            random.shuffle(batch)

        word1_ids = [(word2id[word1] if word1 in word2id else word2id[unk_token]) for label, word1, word2, feat in batch]
        word2_ids = [(word2id[word2] if word2 in word2id else word2id[unk_token]) for label, word1, word2, feat in batch]
        features = [feat for label, word1, word2, feat in batch]
        labels = [label for label, word1, word2, feat in batch]

        if is_testing == True:
            cost, scores = model.test(word1_ids, word2_ids, features, labels)
        elif is_pretraining == True:
            cost, scores = model.pretrain(word1_ids, word2_ids, features, labels, config["learningrate"])
        else:
            cost, scores = model.train(word1_ids, word2_ids, features, labels, config["learningrate"])

        assert(math.isnan(cost) == False and math.isinf(cost) == False), "Cost is "+str(cost) + ", exiting."

        cost_sum += cost

        for x in scores:
            all_predicted_scores.append(x)
        for x in labels:
            all_gold_labels.append(x)

        gc.collect()

    results = evaluate(cost_sum, all_predicted_scores, all_gold_labels, name)
    for key in results:
        print(key + ": " + str(results[key]))

    return results



def run_experiment(config_path):
    config = config_parser.parse_config("config", config_path)
    random.seed(config["random_seed"] if "random_seed" in config else 123)
    temp_model_path = config_path + ".model"

    if "load" in config and config["load"] is not None and len(config["load"]) > 0:
        model = WordPairClassifier.load(config["load"])
        data_test = read_dataset(config["path_test"])
        word2id = model.config["word2id"]
        config = model.config
        process_dataset(data_test, model, word2id, True, config, "test")
        sys.exit()

    data_train = read_dataset(config["path_train"])
    data_dev = read_dataset(config["path_dev"])
    data_test = read_dataset(config["path_test"])
    data_pretrain = read_dataset(config["path_pretrain"]) if ("path_pretrain" in config and config["path_pretrain"] != None and len(config["path_pretrain"]) > 0) else []

    embedding_vocab_set = construct_embedding_vocab([config["word_embedding_path_a"], config["word_embedding_path_b"]])

#    if len(embedding_vocab_set) > 0 and len(data_pretrain) > 0:
#        data_pretrain = [x for x in data_pretrain if (x[1] in embedding_vocab_set and x[2] in embedding_vocab_set)]

    vocabulary = construct_vocabulary([data_train, data_dev, data_test, data_pretrain], embedding_vocab_set if config["restrict_to_embedded_vocab"] == True else None)
    if "extend_vocabulary" in config and config["extend_vocabulary"] == True:
        extend_vocabulary(vocabulary, config["word_embedding_path_a"], config["word_embedding_path_b"], "\t", True)
    word2id = collections.OrderedDict()
    for i in range(len(vocabulary)):
        word2id[vocabulary[i]] = i
    assert(len(word2id) == len(set(vocabulary)))
    config["n_words"] = len(vocabulary)
    config["word2id"] = word2id
    config["feature_count"] = len(data_train[0][3])

    model = WordPairClassifier(config)
    load_embeddings_into_matrix(config["word_embedding_path_a"], model.word_embedding_matrix_A, word2id)
    if config["late_fusion"] == True and config["word_embedding_size_b"] > 0 and config["word_embedding_path_b"] != None and len(config["word_embedding_path_b"]) > 0:
        load_embeddings_into_matrix(config["word_embedding_path_b"], model.word_embedding_matrix_B, word2id)

    for key, val in config.items():
        if key not in ["word2id"]:
            print(str(key) + ": " + str(val))

    if len(data_pretrain) > 0:
        for epoch in range(config["pretrain_epochs"]):
            print("pretrain_epoch: " + str(epoch))
            results_pretrain = process_dataset(data_pretrain, model, word2id, False, True, config, "pretrain")

    best_score = 0.0
    for epoch in range(config["epochs"]):
        print("epoch: " + str(epoch))
        results_train = process_dataset(data_train, model, word2id, False, False, config, "train")
        results_dev = process_dataset(data_dev, model, word2id, True, False, config, "dev")
        score_dev = results_dev[config["model_selector"]]

        if epoch == 0 or score_dev > best_score:
            best_epoch = epoch
            best_score = score_dev
            model.save(temp_model_path)
        print("best_epoch: " + str(best_epoch))
        print("best_measure: " + str(best_score))

        if config["stop_if_no_improvement_for_epochs"] > 0 and (epoch - best_epoch) >= config["stop_if_no_improvement_for_epochs"]:
            break

    if os.path.isfile(temp_model_path):
        model = WordPairClassifier.load(temp_model_path)
        os.remove(temp_model_path)

    if "save" in config and config["save"] is not None and len(config["save"]) > 0:
        model.save(config["save"])

    score_dev = process_dataset(data_dev, model, word2id, True, False, config, "dev_final")
    score_test = process_dataset(data_test, model, word2id, True, False, config, "test")


if __name__ == "__main__":
    run_experiment(sys.argv[1])
