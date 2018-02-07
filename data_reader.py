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
        self.type_of_loss = params_dict['type_of_loss']
        self.sampling = params_dict['sampling']
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
        self.subset_len_train = params_dict.get("subset_len_train")
        self.subset_len_valid = params_dict.get("subset_len_valid")
        self.subset_len_test = params_dict.get("subset_len_test")
        self.seed = params_dict.get("seed")
        self.padding = params_dict.get("padding")
        self.truncating = params_dict.get("truncating")

        np.random.seed(self.seed)

        #nltk.download('punkt')

        if self.dataset_name == 'insurance':
            self.read_data_insurance()
        if self.dataset_name == 'insurance_v1':
            self.read_data_insurance_v1()
        elif self.dataset_name == 'twitter':
                self.read_data_twitter()
        elif self.dataset_name == 'ubuntu':
                    self.read_data_ubuntu()

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

    def read_data_ubuntu(self):
        self.preprocess_data_ubuntu()
        self.preprocess_data_ubuntu("valid")
        self.preprocess_data_ubuntu("test")
        self.subset_data()
        self.subset_data("valid")
        self.subset_data("test")
        self.embdict.create_embedding_matrix()
        self.calculate_steps()

    def preprocess_data_ubuntu(self, data_type="train"):
        fname = self.raw_data_path + 'ubuntu/' + data_type + '.csv'
        questions = []
        answers = []
        labels = []
        candidates = []
        with open(fname) as g:
            tsv_reader = csv.reader(g, delimiter=',')
            for row in tsv_reader:
                questions.append(row[0].split(' '))
                answers.append(row[1].split(' '))
                if data_type == "train":
                    labels.append(row[2])
                else:
                    candidates.append([el.split(' ') for el in row[2:]])
        questions = questions[1:]
        answers = answers[1:]
        self.embdict.add_items(questions)
        self.embdict.add_items(answers)
        if data_type == "train":
            labels = labels[1:]
            self.train_data = [{"id": el[0], "context": el[1][0], "response": el[1][1], "label": int(el[1][2])}
                           for el in enumerate(zip(questions[:-10 * self.batch_size],
                                                   answers[:-10 * self.batch_size], labels[:-10 * self.batch_size]))]
            self.valid_data_train = [{"id": el[0], "context": el[1][0], "response": el[1][1], "label": int(el[1][2])}
                           for el in enumerate(zip(questions[-10 * self.batch_size:],
                                                   answers[-10 * self.batch_size:], labels[-10 * self.batch_size:]))]
        else:
            candidates = candidates[1:]
            for el in candidates:
                self.embdict.add_items(el)
            if data_type == "valid":
                self.valid_data = [{"id": el[0], "context": el[1][0], "response": el[1][1]}
                                   for el in enumerate(zip(questions, answers))]
                self.ranking_samples_pool_valid = candidates
            else:
                if data_type == "test":
                    self.test_data = [{"id": el[0], "context": el[1][0], "response": el[1][1]}
                                      for el in enumerate(zip(questions, answers))]
                    self.ranking_samples_pool_test = candidates

    def subset_data(self, data_type="train"):
        if data_type == "train" and self.subset_len_train is not None:
            ids = np.random.randint(len(self.train_data), size=self.subset_len_train)
            self.train_data = [self.train_data[el] for el in ids]
        elif data_type == "valid" and self.subset_len_valid is not None:
            ids = np.random.randint(len(self.valid_data), size=self.subset_len_valid)
            self.valid_data = [self.valid_data[el] for el in ids]
        elif data_type == "test" and self.subset_len_test is not None:
            ids = np.random.randint(len(self.test_data), size=self.subset_len_test)
            self.test_data = [self.test_data[el] for el in ids]

    def read_data_twitter(self):
        with open(self.raw_data_path + 'twitter/test_twitter_db_m2_maxlen30_mint.txt', 'r') as f:
            lines = f.readlines()
        lists = [ast.literal_eval(el) for el in lines]
        questions = [el[0]['text'].split(' ') for el in lists]
        answers = [el[1]['text'].split(' ') for el in lists]
        self.embdict.add_items(questions)
        self.embdict.add_items(answers)
        self.train_data = [{"id": el[0], "context": el[1][0], "response": el[1][1]}
                           for el in enumerate(zip(questions[:-10 * self.batch_size], answers[:-10 * self.batch_size]))]
        self.valid_data = [{"id": el[0], "context": el[1][0], "response": el[1][1]}
                           for el in enumerate(zip(questions[-10 * self.batch_size:], answers[-10 * self.batch_size:]))]
        self.valid_data_train = self.valid_data
        questions = []
        answers = []
        with open(self.raw_data_path + 'twitter/context-free-testset.tsv') as g:
            tsv_reader = csv.reader(g, delimiter='\t')
            for row in tsv_reader:
                questions.append(row[0])
                answers.append(row[1])
        questions = self.tokenize(questions)
        answers = self.tokenize(answers)
        self.embdict.add_items(questions)
        self.embdict.add_items(answers)
        self.test_data = [{"id": el[0], "context": el[1][0], "response": el[1][1]}
                          for el in enumerate(zip(questions, answers))]
        self.embdict.create_embedding_matrix()
        self.calculate_steps()

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

    def build_label2tokens_vocabulary_insurance(self):
        self.label2tokens_vocab = {}
        answers = []
        for el in self.label2idx_vocab.items():
            self.label2tokens_vocab[el[0]] = self.idx2tokens_insurance(el[1])
            answers.append(self.idx2tokens_insurance(el[1]))
        self.presence_label2tokens = True
        self.embdict.add_items(answers)

    def idx2tokens_insurance(self, utterance_idx):
        utterance_tokens = [self.vocabulary[idx] for idx in utterance_idx]
        return utterance_tokens

    def preprocess_data_insurance(self, data_type="train"):
        file_path = self.raw_data_path + 'InsuranceQA.question.anslabel.token.500.pool.solr.' +\
                    data_type + '.encoded'
        questions = []
        pos_answers = []
        neg_answers = []
        with open(file_path, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            _, q, pa, na = eli.split('\t')
            pa_list = pa.split(' ')
            for elj in pa_list:
                questions.append(q.split(' '))
                pos_answers.append(elj)
                nas = [el for el in na.split(' ') if el != elj]
                neg_answers.append(nas[:499])
        questions = [self.idx2tokens_insurance(el) for el in questions]
        pos_answers = [self.label2tokens_vocab[el] for el in pos_answers]
        self.embdict.add_items(questions)

        data = [{"id": el[0], "context": el[1][0], "response": el[1][1]}
                for el in enumerate(zip(questions, pos_answers))]

        if data_type == "train":
            self.negative_samples_pool = neg_answers
            self.train_data = data[:-10 * self.batch_size]
            self.valid_data_train = data[-10 * self.batch_size:]
        elif data_type == "valid":
                self.ranking_samples_pool_valid = neg_answers
                self.valid_data = data
        elif data_type == "test":
                self.ranking_samples_pool_test = neg_answers
                self.test_data = data

    def read_data_insurance(self):
        self.build_vocabulary_insurance('vocabulary')
        self.build_label2idx_vocabulary_insurance('InsuranceQA.label2answer.token.encoded')
        self.build_label2tokens_vocabulary_insurance()
        self.preprocess_data_insurance('train')
        self.preprocess_data_insurance('valid')
        self.preprocess_data_insurance('test')
        self.embdict.save_items()
        self.embdict.create_embedding_matrix()
        self.subset_data('train')
        self.subset_data('valid')
        self.subset_data('test')
        self.calculate_steps()

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
        pos_answers = [self.label2tokens_vocab[el] for el in pos_answers]
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
        pos_answers = [self.label2tokens_vocab[el] for el in pos_answers]
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
        self.build_label2tokens_vocabulary_insurance()
        self.preprocess_data_insurance_v1_train()
        self.preprocess_data_insurance_v1_valid_test('dev')
        self.preprocess_data_insurance_v1_valid_test('test1')
        self.preprocess_data_insurance_v1_valid_test('test2')
        self.embdict.save_items()
        self.embdict.create_embedding_matrix()
        self.subset_data('train')
        self.subset_data('valid')
        self.subset_data('test')
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

    def make_embeddings(self, sen_list):
        if sen_list is None:
            return None
        embeddings_batch = []
        for sen in sen_list:
            embeddings = []
            for tok in sen:
                embeddings.append(self.embdict.tok2emb.get(tok))
            if len(sen) < self.max_sequence_length:
                pads = [np.zeros(self.embedding_dim) for _ in range(self.max_sequence_length - len(sen))]
                if self.padding == "pre" or self.padding is None:
                    embeddings = pads + embeddings
                elif self.padding == "post":
                    embeddings = embeddings + pads
            else:
                if self.truncating== "pre" or self.truncating is None:
                    embeddings = embeddings[-self.max_sequence_length:]
                elif self.truncating == "post":
                    embeddings = embeddings[:self.max_sequence_length]
            embeddings_batch.append(embeddings)
        embeddings_batch = np.asarray(embeddings_batch)
        return embeddings_batch

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
        y1 = np.ones(self.batch_size)
        if data_type == "train":
            data = self.train_data
            np.random.shuffle(data)
            steps = self.train_steps
        elif data_type == "valid":
            data = self.valid_data_train
            steps = self.valid_steps_train
        # while 1:
        for i in range(steps):
            context_response_data = data[i * self.batch_size:(i + 1) * self.batch_size]
            context_response_indices = [el["id"] for el in context_response_data]
            context_data = [el["context"] for el in context_response_data]
            response_data = [el["response"] for el in context_response_data]
            negative_response_data = None

            if self.sampling == "no_sampling":
                if self.type_of_loss == "triplet_hinge":
                    negative_response_data = [el["negative_response"] for el in context_response_data]
                    context = self.make_integers(context_data)
                    response = self.make_integers(response_data)
                    negative_response = self.make_integers(negative_response_data)
                    yield ([context, response, negative_response], y0)
                elif self.type_of_loss == "binary_crossentropy":
                    context = self.make_integers(context_data)
                    response = self.make_integers(response_data)
                    y = np.asarray([el["label"] for el in context_response_data])
                    yield ([context, response], y)

            if self.sampling == "random":
                negative_response_data = self.create_neg_resp_rand(context_response_indices, data_type)
            elif self.sampling == "negative":
                negative_response_data = self.create_neg_resp_ns(context_response_indices, data_type)
            elif self.sampling == "negative_with_context":
                negative_response_data = self.create_neg_resp_nswc(context_response_indices, data_type)

            if negative_response_data is not None:
                if self.type_of_loss == "triplet_hinge":
                    context = self.make_integers(context_data)
                    response = self.make_integers(response_data)
                    negative_response = self.make_integers(negative_response_data)
                    yield ([context, response, negative_response], y0)
                elif self.type_of_loss == "binary_crossentropy":
                    batch_data = [list(el) for el in zip(context_data, response_data, y1)] +\
                                 [list(el) for el in zip(context_data, negative_response_data, y0)]
                    np.random.shuffle(batch_data)
                    context_data = [el[0] for el in batch_data]
                    response_data = [el[1] for el in batch_data]
                    y = [el[2] for el in batch_data]
                    context = self.make_integers(context_data)
                    response = self.make_integers(response_data)
                    y = np.asarray(y)
                    yield ([context, response], y)

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
                negative_response_data = [self.label2tokens_vocab[el] for el in candidate_numbers]
            else:
                negative_response_data = candidate_numbers
        elif sample_candidates == "batch" or sample_candidates is None:
            candidate_lists = self.batch_size * [data_indices]
            candidate_indices = np.random.randint(0, self.batch_size, self.batch_size)
            candidate_numbers = [candidate_lists[i][candidate_indices[i]] for i in range(self.batch_size)]
            negative_response_data = [el["response"] for el in data if el["id"] in candidate_numbers]
        elif sample_candidates == "global":
            candidate_indices = []
            for i in range(self.batch_size):
                candidate_index = np.random.randint(1, len(self.label2tokens_vocab) + 1, 1)[0]
                while str(candidate_index) in positive_answers_pool[data_indices[i]]:
                    candidate_index = np.random.randint(1, len(self.label2tokens_vocab) + 1, 1)[0]
                candidate_indices.append(str(candidate_index))
            negative_response_data = [self.label2tokens_vocab[el] for el in candidate_indices]
        return negative_response_data

    def create_neg_resp_ns(self, data_indices, data_type):
        if data_type == "train":
            data = self.train_data
            negative_samples_pool = self.negative_samples_pool
        else:
            if data_type == "valid":
                data = self.valid_data_train
                negative_samples_pool = self.negative_samples_pool_valid_train
        context_data = []
        response_data = []
        candidate_data = []
        negative_response_data = []
        scores = []
        if negative_samples_pool is not None:
            for i in range(self.num_negative_samples):
                context = [el["context"] for el in data if el["id"] in data_indices]
                context_data += [context]
                response = [el["response"] for el in data if el["id"] in data_indices]
                response_data += [response]
                candidate_indices = [negative_samples_pool[el][i] for el in data_indices]
                if self.presence_label2tokens:
                    candidate = [self.label2tokens_vocab[el] for el in candidate_indices]
                else:
                    candidate = candidate_indices
                candidate_data += [candidate]
            for i in range(self.num_negative_samples):
                context_batch = self.make_integers(context_data[i])
                response_batch = self.make_integers(response_data[i])
                candidate_batch = self.make_integers(candidate_data[i])
                s_pos = self.score_model.predict_on_batch([context_batch, response_batch])
                s_neg = self.score_model.predict_on_batch([context_batch, candidate_batch])
                scores.append(s_pos - s_neg)
            scores = np.hstack(scores)

            for i in range(self.batch_size):
                valid_indices = []
                for j in range(self.num_negative_samples):
                    if scores[i][j] <= self.margin:
                        valid_indices.append(j)
                if len(valid_indices) == 0:
                    print('There is no valid examples in the current batch.')
                    return None
                else:
                    index = np.random.choice(valid_indices)
                    response = candidate_data[index][i]
                    negative_response_data.append(response)
        else:
            for i in range(self.batch_size):
                context = (self.batch_size - 1) * [el["context"] for el in data if el["id"] == data_indices[i]]
                context_data += context
                response = (self.batch_size - 1) * [el["response"] for el in data if el["id"] == data_indices[i]]
                response_data += response
                candidate = [el["response"] for el in data if el["id"] != data_indices[i]]
                candidate_data += candidate

            context_data = [context_data[i:-1:(self.batch_size - 1)] for i in range(self.batch_size-1)]
            response_data = [response_data[i:-1:(self.batch_size - 1)] for i in range(self.batch_size-1)]
            candidate_data = [candidate_data[i:-1:(self.batch_size - 1)] for i in range(self.batch_size-1)]

            for i in range(self.batch_size - 1):
                context_batch = self.make_integers(context_data[i])
                response_batch = self.make_integers(response_data[i])
                candidate_batch = self.make_integers(candidate_data[i])
                s_pos = self.score_model.predict_on_batch([context_batch, response_batch])
                s_neg = self.score_model.predict_on_batch([context_batch, candidate_batch])
                scores.append(s_pos - s_neg)
            scores = np.hstack(scores)

            for i in range(self.batch_size):
                valid_indices = []
                for j in range(self.batch_size - 1):
                    if scores[i][j] <= self.margin:
                        valid_indices.append(j)
                if len(valid_indices) == 0:
                    print('There is no valid examples in the current batch.')
                    return None
                else:
                    index = np.random.choice(valid_indices)
                    response = candidate_data[index][i]
                    negative_response_data.append(response)
        return negative_response_data

    def create_neg_resp_nswc(self, data_indices, data_type):
        if data_type == "train":
            data = self.train_data
            negative_samples_pool = self.negative_samples_pool
        else:
            if data_type == "valid":
                data = self.valid_data_train
                negative_samples_pool = self.negative_samples_pool_valid_train
        context_data = []
        response_data = []
        candidate_data = []
        negative_response_data = []
        scores = []
        if negative_samples_pool is not None:
            for i in range(self.num_negative_samples):
                context = [el["context"] for el in data if el["id"] in data_indices]
                context_data += [context]
                response = [el["response"] for el in data if el["id"] in data_indices]
                response_data += [response]
                candidate_indices = [negative_samples_pool[el][i] for el in data_indices]
                if self.presence_label2tokens:
                    candidate = [self.label2tokens_vocab[el] for el in candidate_indices]
                else:
                    candidate = candidate_indices
                candidate_data += [candidate]
            context_data += [context]
            response_data += [response]
            candidate_data += [context]
            for i in range(self.num_negative_samples + 1):
                context_batch = self.make_integers(context_data[i])
                response_batch = self.make_integers(response_data[i])
                candidate_batch = self.make_integers(candidate_data[i])
                s_pos = self.score_model.predict_on_batch([context_batch, response_batch])
                s_neg = self.score_model.predict_on_batch([context_batch, candidate_batch])
                scores.append(s_pos - s_neg)
            scores = np.hstack(scores)
            for i in range(self.batch_size):
                valid_indices = []
                for j in range(self.num_negative_samples + 1):
                    if scores[i][j] <= self.margin:
                        valid_indices.append(j)
                if len(valid_indices) == 0:
                    print('There is no valid examples in the current batch.')
                    return None
                else:
                    index = np.random.choice(valid_indices)
                    response = candidate_data[index][i]
                    negative_response_data.append(response)

        else:
            for i in range(self.batch_size):
                context = self.batch_size * [el["context"] for el in data if el["id"] == data_indices[i]]
                context_data += context
                response = self.batch_size * [el["response"] for el in data if el["id"] == data_indices[i]]
                response_data += response
                candidate = [el["response"] for el in data if el["id"] != data_indices[i]] + \
                            [el["context"] for el in data if el["id"] == data_indices[i]]
                candidate_data += candidate

            context_data = [context_data[i:-1:self.batch_size] for i in range(self.batch_size)]
            response_data = [response_data[i:-1:self.batch_size] for i in range(self.batch_size)]
            candidate_data = [candidate_data[i:-1:self.batch_size] for i in range(self.batch_size)]

            for i in range(self.batch_size):
                context_batch = self.make_integers(context_data[i])
                response_batch = self.make_integers(response_data[i])
                candidate_batch = self.make_integers(candidate_data[i])
                s_pos = self.score_model.predict_on_batch([context_batch, response_batch])
                s_neg = self.score_model.predict_on_batch([context_batch, candidate_batch])
                scores.append(s_pos - s_neg)
            scores = np.hstack(scores)

            for i in range(self.batch_size):
                valid_indices = []
                for j in range(self.batch_size):
                    if scores[i][j] <= self.margin:
                        valid_indices.append(j)
                if len(valid_indices) == 0:
                    print('There is no valid examples in the current batch.')
                    return None
                else:
                    index = np.random.choice(valid_indices)
                    response = candidate_data[index][i]
                    negative_response_data.append(response)
        return negative_response_data

    def batch_generator_test(self, data_type="valid"):
        if data_type == "valid":
            print("valid generator")
            #np.random.shuffle(self.valid_data)
            data = self.valid_data
            num_steps = self.valid_steps
            batch_size = self.val_batch_size
        else:
            if data_type == "test":
                print("test generator")
                #np.random.shuffle(self.test_data)
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
                    response = [self.label2tokens_vocab[el] for el in response_indices]
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
