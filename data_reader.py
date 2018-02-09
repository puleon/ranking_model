import numpy as np
import ast
import csv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import json
from keras.preprocessing.sequence import pad_sequences

from embeddings_dict import EmbeddingsDict


class DataReader(object):

    def __init__(self, score_model, params_dict):
        self.score_model = score_model
        self.margin = params_dict['margin']
        self.sample_candidates = params_dict.get('sample_candidates')
        self.sample_candidates_valid = params_dict.get('sample_candidates_valid')
        self.sample_candidates_test = params_dict.get('sample_candidates_test')
        self.num_negative_samples = params_dict['num_negative_samples']
        self.num_ranking_samples_valid = params_dict['num_ranking_samples_valid']
        self.num_ranking_samples_test = params_dict['num_ranking_samples_test']
        self.raw_data_path = params_dict['raw_dataset_path']
        self.batch_size = params_dict['batch_size']
        self.max_sequence_length = params_dict['max_sequence_length']
        self.embedding_dim = params_dict['embedding_dim']
        self.embdict = EmbeddingsDict(params_dict)
        self.dataset_name = params_dict['dataset_name']
        self.val_batch_size = params_dict.get("val_batch_size")
        self.test_batch_size = params_dict.get("test_batch_size")
        self.seed = params_dict.get("seed")
        self.padding = params_dict.get("padding")
        self.truncating = params_dict.get("truncating")

        np.random.seed(self.seed)

        self.label2context_vocab = {}
        self.label2response_vocab = {}


        #nltk.download('punkt')

        if self.dataset_name == 'insurance_v1':
            self.read_data_insurance_v1()

        if self.sample_candidates_valid == 'global':
                self.num_ranking_samples_valid = len(self.label2response_vocab)
        if self.sample_candidates_test == 'global':
                self.num_ranking_samples_test = len(self.label2response_vocab)

        print('Length of train data:', len(self.train_data))
        print('Length of validation data for train:', len(self.valid_data_train))
        print('Length of validation data:', len(self.valid_data))
        print('Length of test data:', len(self.test_data))

    def tokenize(self, sen_list):
        sen_tokens_list = []
        for sen in sen_list:
            sent_toks = sent_tokenize(sen)
            word_toks = [word_tokenize(el) for el in sent_toks]
            tokens = [val for sublist in word_toks for val in sublist]
            tokens = [el for el in tokens if el != '']
            sen_tokens_list.append(tokens)
        return sen_tokens_list

    def build_vocabulary_insurance(self, fname):
        with open(self.raw_data_path + fname) as f:
            data = f.readlines()
            self.vocabulary = {el.split('\t')[0]: el.split('\t')[1][:-1] for el in data}

    def build_label2idx_vocabulary_insurance(self, fname):
        with open(self.raw_data_path + fname, 'r') as f:
            data = f.readlines()
            self.label2idx_vocab = {int(el.split('\t')[0])-1: (el.split('\t')[1][:-1]).split(' ') for el in data}

    def build_label2response_vocabulary_insurance(self):
        responses = []
        for el in self.label2idx_vocab.items():
            self.label2response_vocab[el[0]] = self.idx2tokens_insurance(el[1])
            responses.append(self.idx2tokens_insurance(el[1]))
        self.embdict.add_items(responses)

    def context2label_insurance(self, contexts):
        self.embdict.add_items(contexts)
        if self.label2context_vocab == {}:
            id = 0
        else:
            id = np.max(list(self.label2context_vocab.keys())) + 1
        for i in range(len(contexts)):
            self.label2context_vocab[id+i] = contexts[i]
        return np.arange(id, len(contexts) + id)

    def idx2tokens_insurance(self, utterance_idx):
        utterance_tokens = [self.vocabulary[idx] for idx in utterance_idx]
        return utterance_tokens

    def preprocess_data_insurance_v1_train(self):
        positive_responses_pool = []
        file_path = self.raw_data_path + 'question.train.token_idx.label'
        contexts = []
        responses = []
        with open(file_path, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            q, pa = eli.split('\t')
            pa_list = [int(el)-1 for el in pa.split(' ')]
            for elj in pa_list:
                contexts.append(q.split(' '))
                responses.append(elj)
                positive_responses_pool.append(pa_list)
        contexts = [self.idx2tokens_insurance(el) for el in contexts]
        contexts = self.context2label_insurance(contexts)

        self.train_data = [{"context": el[0], "response": el[1], "pos_pool": el[2], "neg_pool": None}
                           for el in zip(contexts, responses, positive_responses_pool)]

    def preprocess_data_insurance_v1_valid_test(self, data_type="dev"):
        file_path = self.raw_data_path + 'question.' + data_type + '.label.token_idx.pool'
        pos_responses_pool = []
        neg_responses_pool = []
        contexts = []
        pos_responses = []
        with open(file_path, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            pa, q, na = eli.split('\t')
            pa_list = [int(el)-1 for el in pa.split(' ')]
            for elj in pa_list:
                contexts.append(q.split(' '))
                pos_responses.append(elj)
                pos_responses_pool.append(pa_list)
                nas = [int(el)-1 for el in na.split(' ')]
                nas = [el for el in nas if el not in pa_list]
                neg_responses_pool.append(nas)
        contexts = [self.idx2tokens_insurance(el) for el in contexts]
        contexts = self.context2label_insurance(contexts)

        data = [{"context": el[0], "response": el[1], "pos_pool": el[2], "neg_pool": el[3]}
                for el in zip(contexts, pos_responses, pos_responses_pool, neg_responses_pool)]

        if data_type == "dev":
            self.valid_data_train = data
        elif data_type == "test1":
            self.valid_data = data
        elif data_type == "test2":
            self.test_data = data

    def read_data_insurance_v1(self):
        self.build_vocabulary_insurance('vocabulary')
        self.build_label2idx_vocabulary_insurance('answers.label.token_idx')
        self.build_label2response_vocabulary_insurance()
        self.preprocess_data_insurance_v1_train()
        self.preprocess_data_insurance_v1_valid_test('dev')
        self.preprocess_data_insurance_v1_valid_test('test1')
        self.preprocess_data_insurance_v1_valid_test('test2')
        self.embdict.save_items()
        self.embdict.create_embedding_matrix()
        self.calculate_steps()

    def calculate_steps(self):
        if self.val_batch_size is None:
            self.val_batch_size = len(self.valid_data)
        if self.test_batch_size is None:
            self.test_batch_size = len(self.test_data)
        self.train_steps = len(self.train_data) // self.batch_size
        self.valid_steps_train = len(self.valid_data_train) // self.batch_size
        self.valid_steps = len(self.valid_data) // self.val_batch_size
        self.test_steps = len(self.test_data) // self.test_batch_size

    def make_integers(self, labels_list, type):
        if type == "context":
            vocab = self.label2context_vocab
        elif type == "response":
            vocab = self.label2response_vocab
        if labels_list is None:
            return None
        integers_list = []
        for label in labels_list:
            integers = []
            sen = vocab[label]
            for tok in sen:
                integers.append(self.embdict.tok_index[tok])
            integers_list.append(integers)
        integers_list = pad_sequences(integers_list, maxlen=self.max_sequence_length,
                                      padding=self.padding, truncating=self.truncating)
        return integers_list

    def get_model(self, model):
        self.score_model = model

    def batch_generator_train(self, data_type="train"):
        y0 = np.zeros(self.batch_size)
        if data_type == "train":
            data = self.train_data
            np.random.shuffle(data)
            steps = self.train_steps
        elif data_type == "valid":
            data = self.valid_data_train
            steps = self.valid_steps_train
        for i in range(steps):
            context_response_data = data[i * self.batch_size:(i + 1) * self.batch_size]
            context_data = [el["context"] for el in context_response_data]
            response_data = [el["response"] for el in context_response_data]
            negative_response_data = self.create_neg_resp_rand(context_response_data, data_type)

            if negative_response_data is not None:
                context = self.make_integers(context_data, "context")
                response = self.make_integers(response_data, "response")
                negative_response = self.make_integers(negative_response_data, "response")
                yield ([context, response, negative_response], y0)

    def create_neg_resp_rand(self, context_response_data, data_type):
        if data_type == "train":
            sample_candidates = self.sample_candidates
        elif data_type == "valid":
            sample_candidates = self.sample_candidates_valid
        if sample_candidates == "pool":
            candidate_lists = [el["neg_pool"] for el in context_response_data]
            candidate_indices = [np.random.randint(0, np.min([len(candidate_lists[i]),
                                 self.num_negative_samples]), 1)[0]
                                 for i in range(self.batch_size)]
            negative_response_data = [candidate_lists[i][candidate_indices[i]] for i in range(self.batch_size)]
        elif sample_candidates == "global":
            candidates = []
            for i in range(self.batch_size):
                candidate = np.random.randint(0, len(self.label2response_vocab), 1)[0]
                while candidate in context_response_data[i]["pos_pool"]:
                    candidate = np.random.randint(0, len(self.label2response_vocab), 1)[0]
                candidates.append(candidate)
            negative_response_data = candidates
        return negative_response_data

    def batch_generator_test(self, data_type="valid"):
        if data_type == "valid":
            print("valid generator")
            data = self.valid_data
            num_steps = self.valid_steps
            batch_size = self.val_batch_size
        else:
            if data_type == "test":
                print("test generator")
                data = self.test_data
                num_steps = self.test_steps
                batch_size = self.test_batch_size
        for i in range(num_steps + 1):
            if i < num_steps:
                context_response_data = data[i * batch_size:(i + 1) * batch_size]
            else:
                context_response_data = data[i * batch_size:len(data)]
            context_data = [el["context"] for el in context_response_data]
            context = self.make_integers(context_data, "context")
            response_data, y = self.create_rank_resp(context_response_data, data_type)
            for el in zip(response_data, y):
                response = self.make_integers(el[0], "response")
                yield ([context, response], el[1])

    def create_rank_resp(self, context_response_data, data_type="valid"):
        if data_type == "valid":
            ranking_length = self.num_ranking_samples_valid
            sample_candidates = self.sample_candidates_valid
        elif data_type == "test":
            ranking_length = self.num_ranking_samples_test
            sample_candidates = self.sample_candidates_test
        if sample_candidates == "pool":
            y = ranking_length * [np.array([len(el["pos_pool"]) for el in context_response_data])]
            response_data = []
            for i in range(len(context_response_data)):
                pos_pool = context_response_data[i]["pos_pool"]
                neg_pool = context_response_data[i]["neg_pool"]
                response = pos_pool + neg_pool
                response_data += response[:ranking_length]
            response_data = [response_data[i::ranking_length] for i in range(ranking_length)]

        elif sample_candidates == "global" or sample_candidates is None:
            y = ranking_length * [np.array([len(el["pos_pool"]) for el in context_response_data])]
            response_data = []
            for i in range(len(context_response_data)):
                pos_pool = context_response_data[i]["pos_pool"]
                neg_pool = [el for el in list(self.label2response_vocab.keys())
                            if el not in context_response_data[i]["pos_pool"]]
                response = pos_pool + neg_pool
                response_data += response
            response_data = [response_data[i::ranking_length] for i in range(ranking_length)]
        return response_data, y
