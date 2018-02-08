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
        self.raw_data_path = params_dict['raw_dataset_path']
        self.batch_size = params_dict['batch_size']
        self.max_sequence_length = params_dict['max_sequence_length']
        self.embedding_dim = params_dict['embedding_dim']
        self.embdict = EmbeddingsDict(params_dict)
        self.dataset_name = params_dict['dataset_name']
        self.negative_samples_pool = None
        self.num_negative_samples = params_dict['num_negative_samples']
        self.ranking_samples_pool_valid = None
        self.num_ranking_samples_valid = params_dict['num_ranking_samples_valid']
        self.ranking_samples_pool_test = None
        self.num_ranking_samples_test = params_dict['num_ranking_samples_test']
        self.val_batch_size = params_dict.get("val_batch_size")
        self.test_batch_size = params_dict.get("test_batch_size")
        self.presence_label2tokens = False
        self.seed = params_dict.get("seed")
        self.padding = params_dict.get("padding")
        self.truncating = params_dict.get("truncating")

        np.random.seed(self.seed)

        #nltk.download('punkt')

        if self.dataset_name == 'insurance_v1':
            self.read_data_insurance_v1()

        if self.sample_candidates_valid == 'global':
                self.num_ranking_samples_valid = len(self.valid_data)
        if self.sample_candidates_test == 'global':
                self.num_ranking_samples_test = len(self.test_data)

        print('Length of train data:', len(self.train_data))
        if self.negative_samples_pool is not None:
            print('Negative samples pool shape:',
                  (len(self.negative_samples_pool), len(self.negative_samples_pool[-1])))
        print('Length of validation data for train:', len(self.valid_data_train))
        print('Length of validation data:', len(self.valid_data))
        if self.ranking_samples_pool_valid is not None:
            print('Ranking validation samples pool shape:',
                  (len(self.ranking_samples_pool_valid), len(self.ranking_samples_pool_valid[-1])))
        print('Length of test data:', len(self.test_data))
        if self.ranking_samples_pool_test is not None:
            print('Ranking test samples pool shape:',
                  (len(self.ranking_samples_pool_test), len(self.ranking_samples_pool_test[-1])))

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
            self.label2idx_vocab = {el.split('\t')[0]: (el.split('\t')[1][:-1]).split(' ') for el in data}

    def build_label2response_vocabulary_insurance(self):
        self.label2response_vocab = {}
        answers = []
        for el in self.label2idx_vocab.items():
            self.label2response_vocab[el[0]] = self.idx2tokens_insurance(el[1])
            answers.append(self.idx2tokens_insurance(el[1]))
        self.presence_label2tokens = True
        self.embdict.add_items(answers)

    def build_label2response_vocabulary_insurance(self):
        self.label2response_vocab = {}
        answers = []
        for el in self.label2idx_vocab.items():
            self.label2response_vocab[el[0]] = self.idx2tokens_insurance(el[1])
            answers.append(self.idx2tokens_insurance(el[1]))
        self.presence_label2tokens = True
        self.embdict.add_items(answers)



    def idx2tokens_insurance(self, utterance_idx):
        utterance_tokens = [self.vocabulary[idx] for idx in utterance_idx]
        return utterance_tokens

    def preprocess_data_insurance_v1_train(self):
        self.positive_answers_pool = []
        file_path = self.raw_data_path + 'question.train.token_idx.label'
        questions = []
        pos_answers = []
        with open(file_path, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            q, pa = eli.split('\t')
            pa_list = pa.split(' ')
            for elj in pa_list:
                questions.append(q.split(' '))
                pos_answers.append(elj)
                self.positive_answers_pool.append(pa_list)
        questions = [self.idx2tokens_insurance(el) for el in questions]
        pos_answers = [self.label2response_vocab[el] for el in pos_answers]
        self.embdict.add_items(questions)

        self.train_data = [{"id": el[0], "context": el[1][0], "response": el[1][1]}
                for el in enumerate(zip(questions, pos_answers))]

    def preprocess_data_insurance_v1_valid_test(self, data_type="dev"):
        file_path = self.raw_data_path + 'question.' + data_type + '.label.token_idx.pool'
        positive_answers_pool = []
        questions = []
        pos_answers = []
        neg_answers = []
        with open(file_path, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            pa, q, na = eli.split('\t')
            pa_list = pa.split(' ')
            for elj in pa_list:
                questions.append(q.split(' '))
                pos_answers.append(elj)
                positive_answers_pool.append(pa_list)
                nas = na.split(' ')
                nas.remove(elj)
                neg_answers.append(nas)
        questions = [self.idx2tokens_insurance(el) for el in questions]
        pos_answers = [self.label2response_vocab[el] for el in pos_answers]
        self.embdict.add_items(questions)

        data = [{"id": el[0], "context": el[1][0], "response": el[1][1]}
                for el in enumerate(zip(questions, pos_answers))]

        if data_type == "dev":
            self.positive_answers_pool_valid_train = positive_answers_pool
            self.negative_samples_pool_valid_train = neg_answers
            self.valid_data_train = data
        elif data_type == "test1":
            self.positive_answers_pool_valid = positive_answers_pool
            self.ranking_samples_pool_valid = neg_answers
            self.valid_data = data
        elif data_type == "test2":
            self.positive_answers_pool_test = positive_answers_pool
            self.ranking_samples_pool_test = neg_answers
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

    def make_integers(self, sen_list):
        if sen_list is None:
            return None
        integers_list = []
        for sen in sen_list:
            integers = []
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
            context_response_indices = [el["id"] for el in context_response_data]
            context_data = [el["context"] for el in context_response_data]
            response_data = [el["response"] for el in context_response_data]
            negative_response_data = self.create_neg_resp_rand(context_response_indices, data_type)

            if negative_response_data is not None:
                context = self.make_integers(context_data)
                response = self.make_integers(response_data)
                negative_response = self.make_integers(negative_response_data)
                yield ([context, response, negative_response], y0)

    def create_neg_resp_rand(self, data_indices, data_type):
        if data_type == "train":
            data = self.train_data
            negative_samples_pool = self.negative_samples_pool
            positive_answers_pool = self.positive_answers_pool
            sample_candidates = self.sample_candidates
        elif data_type == "valid":
            data = self.valid_data_train
            negative_samples_pool = self.negative_samples_pool_valid_train
            positive_answers_pool = self.positive_answers_pool_valid_train
            sample_candidates = self.sample_candidates_valid
        if sample_candidates == "pool":
            candidate_lists = [negative_samples_pool[el] for el in data_indices]
            candidate_indices = [np.random.randint(0, np.min([len(candidate_lists[i]),
                                 self.num_negative_samples]), 1)[0]
                                 for i in range(self.batch_size)]
            candidate_numbers = [candidate_lists[i][candidate_indices[i]] for i in range(self.batch_size)]
            if self.presence_label2tokens:
                negative_response_data = [self.label2response_vocab[el] for el in candidate_numbers]
            else:
                negative_response_data = candidate_numbers
        elif sample_candidates == "global":
            candidate_indices = []
            for i in range(self.batch_size):
                candidate_index = np.random.randint(1, len(self.label2response_vocab) + 1, 1)[0]
                while str(candidate_index) in positive_answers_pool[data_indices[i]]:
                    candidate_index = np.random.randint(1, len(self.label2response_vocab) + 1, 1)[0]
                candidate_indices.append(str(candidate_index))
            negative_response_data = [self.label2response_vocab[el] for el in candidate_indices]
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
            data_indices = [el["id"] for el in context_response_data]
            context_data = [el["context"] for el in context_response_data]
            context = self.make_integers(context_data)
            response_data, y, y_set = self.create_rank_resp(data_indices, data_type)
            for el in zip(response_data, y, y_set):
                response = self.make_integers(el[0])
                yield ([context, response], el[1], el[2])

    def create_rank_resp(self, data_indices, data_type="valid"):
        if data_type == "valid":
            data = self.valid_data
            ranking_length = self.num_ranking_samples_valid
            batch_size = self.val_batch_size
            pool = self.ranking_samples_pool_valid
            positive_pool = self.positive_answers_pool_valid
            sample_candidates = self.sample_candidates_valid
        elif data_type == "test":
            data = self.test_data
            ranking_length = self.num_ranking_samples_test
            batch_size = self.test_batch_size
            pool = self.ranking_samples_pool_test
            positive_pool = self.positive_answers_pool_test
            sample_candidates = self.sample_candidates_test
        if sample_candidates == "pool":
            y_set = (ranking_length + 1) * [[np.arange(len(positive_pool[el])) for el in data_indices]]
            y = (ranking_length + 1) * [np.zeros(batch_size)]
            response_data = [[el["response"] for el in data if el['id'] in data_indices]]
            pool_indices = np.arange(ranking_length)
            for i in pool_indices:
                response_indices = [pool[el][i] for el in data_indices]
                if self.presence_label2tokens:
                    response = [self.label2response_vocab[el] for el in response_indices]
                else:
                    response = response_indices
                response_data.append(response)
        elif sample_candidates == "global" or sample_candidates is None:
            y_set = (ranking_length + 1) * [[np.arange(len(positive_pool[el])) for el in data_indices]]
            y = (ranking_length + 1) * [np.zeros(batch_size)]
            response_data = []
            for i in range(batch_size):
                response = [el["response"] for el in data if el["id"] != data_indices[i]]
                response_data += response

            response_data = [response_data[i::(ranking_length-1)] for i in range(ranking_length-1)]
            response_data = [[el["response"] for el in data if el["id"] in data_indices]] + response_data
            response_data = response_data + [[el["context"] for el in data if el["id"] in data_indices]]
        return response_data, y, y_set
