from keras.layers import Input, LSTM, Lambda, Embedding
from keras.layers.merge import Dot
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.initializers import glorot_uniform, Orthogonal
from keras import backend as K
from keras import callbacks
from keras.models import load_model
import os
import numpy as np
from data_reader import DataReader
import custom_metrics
from shutil import copyfile

class RankingModel(object):

    def __init__(self, params_dict):
        self.run_type = params_dict.get("run_type")
        self.device_number = params_dict["device_number"]
        self.load_path = params_dict.get("load_path")
        self.pooling = params_dict.get("pooling")
        self.recurrent = params_dict.get("recurrent")
        self.save_path = params_dict['save_path']
        self.max_sequence_length = params_dict['max_sequence_length']
        self.embedding_dim = params_dict['embedding_dim']
        self.seed = params_dict['seed']
        self.hidden_dim = params_dict['hidden_dim']
        self.learning_rate = params_dict['learning_rate']
        self.margin = params_dict['margin']
        self.type_of_weights = params_dict['type_of_weights']
        self.epoch_num = params_dict["epoch_num"]
        self.epoch_num_valid = params_dict.get("epoch_num_valid")

        self.metrics = ["rank_response", "r_at_1", "r_at_2", "r_at_5"]
        self.metrics_functions = [custom_metrics.rank_response,
                                  custom_metrics.r_at_1,
                                  custom_metrics.r_at_2,
                                  custom_metrics.r_at_5]

        if self.epoch_num_valid is None:
            self.epoch_num_valid = 1

        self.reader = DataReader(params_dict)

        self.loss = self.triplet_loss
        self.optimizer = Adam(lr=self.learning_rate)
        self.obj_model = self.triplet_hinge_loss_model()
        self.obj_model.compile(loss=self.loss, optimizer=self.optimizer)

        self.checkpoint = callbacks.ModelCheckpoint(
            filepath= self.save_path + "/model.hdf5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=1)
        self.checkpoint.set_model(self.obj_model)
        self.csv_losses = callbacks.CSVLogger(self.save_path + '/losses.csv')
        self.csv_losses.set_model(self.obj_model)
        self.csv_valid_metrics = callbacks.CSVLogger(self.save_path + '/valid_metrics.csv')
        self.csv_valid_metrics.set_model(self.obj_model)
        self.csv_test_metrics = callbacks.CSVLogger(self.save_path + '/test_metrics.csv')
        self.csv_test_metrics.set_model(self.obj_model)

        if os.path.isfile(self.load_path + "/model.hdf5"):
            print("The model file exists. Loading the model.")
            self.obj_model.load_weights(self.load_path + "/model.hdf5")
        else:
            os.mkdir(self.save_path)
            copyfile('./config.json', self.save_path + '/config.json')
            #self.score_model = self.obj_model.get_layer(name="score_model")
        self.fit_custom()

    def score_difference(self, inputs):
        """Define a function for a lambda layer of a model."""

        return Lambda(lambda x: x[0] - x[1])(inputs)

    def score_difference_output_shape(self, shapes):
        """Define an output shape of a lambda layer of a model."""

        return shapes[0]

    def max_pooling(self, input):
        """Define a function for a lambda layer of a model."""

        return K.max(input, axis=1, keepdims=False)

    def max_pooling_output_shape(self, shape):
        """Define an output shape of a lambda layer of a model."""

        return shape[0], shape[2]

    def create_embedding_layer(self, input_dim):
        inp = Input(shape=(input_dim,))
        out = Embedding(self.reader.embdict.index,
                        self.embedding_dim,
                        weights=[self.reader.embdict.embedding_matrix],
                        input_length=input_dim,
                        trainable=True)(inp)
        model = Model(inputs=inp, outputs=out, name="word_embedding_model")
        return model


    def create_lstm_layer_max_pooling(self, input_dim):
        """Create a LSTM layer of a model."""
        if self.pooling is None or self.pooling == "max":
            ret_seq = True
        elif self.pooling == "no":
            ret_seq = False
        inp = Input(shape=(input_dim,  self.embedding_dim,))
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        if self.recurrent == "bilstm" or self.recurrent is None:
            out = Bidirectional(LSTM(self.hidden_dim,
                                input_shape=(input_dim, self.embedding_dim,),
                                kernel_initializer=ker_in,
                                recurrent_initializer=rec_in,
                                return_sequences=ret_seq), merge_mode='concat')(inp)
        elif self.recurrent == "lstm":
            out = LSTM(self.hidden_dim,
                       input_shape=(input_dim, self.embedding_dim,),
                       kernel_initializer=ker_in,
                       recurrent_initializer=rec_in,
                       return_sequences=ret_seq)(inp)
        model = Model(inputs=inp, outputs=out)
        return model

    def maxpool_cosine_score_model(self, input_dim):
        """Define a model with bi-LSTM layers and without attention."""

        input_a = Input(shape=(input_dim,))
        input_b = Input(shape=(input_dim,))
        if self.type_of_weights == "shared":
            embedding_layer = self.create_embedding_layer(self.max_sequence_length)
            emb_a = embedding_layer(input_a)
            emb_b = embedding_layer(input_b)
            lstm_layer = self.create_lstm_layer_max_pooling(self.max_sequence_length)
            lstm_a = lstm_layer(emb_a)
            lstm_b = lstm_layer(emb_b)
        elif self.type_of_weights == "separate":
            embedding_layer_a = self.create_embedding_layer(self.max_sequence_length)
            embedding_layer_b = self.create_embedding_layer(self.max_sequence_length)
            emb_a = embedding_layer_a(input_a)
            emb_b = embedding_layer_b(input_b)
            lstm_layer_a = self.create_lstm_layer_max_pooling(self.max_sequence_length)
            lstm_layer_b = self.create_lstm_layer_max_pooling(self.max_sequence_length)
            lstm_a = lstm_layer_a(emb_a)
            lstm_b = lstm_layer_b(emb_b)
        if self.pooling is None or self.pooling == "max":
            lstm_a = Lambda(self.max_pooling, output_shape=self.max_pooling_output_shape,
                                name="max_pooling_a")(lstm_a)
            lstm_b = Lambda(self.max_pooling, output_shape=self.max_pooling_output_shape,
                                name="max_pooling_b")(lstm_b)
        cosine = Dot(normalize=True, axes=-1)([lstm_a, lstm_b])
        model = Model([input_a, input_b], cosine, name="score_model")
        return model

    def triplet_hinge_loss_model(self):
        question = Input(shape=(self.max_sequence_length,))
        answer_positive = Input(shape=(self.max_sequence_length,))
        answer_negative = Input(shape=(self.max_sequence_length,))
        self.score_model = self.maxpool_cosine_score_model(self.max_sequence_length)
        score_positive = self.score_model([question, answer_positive])
        score_negative = self.score_model([question, answer_negative])
        score_diff = Lambda(self.score_difference, output_shape=self.score_difference_output_shape,
                      name="score_diff")([score_positive, score_negative])
        model = Model([question, answer_positive, answer_negative], score_diff)
        return model

    def triplet_loss(self, y_true, y_pred):
        """Triplet loss function"""

        return K.mean(K.maximum(self.margin - y_pred, 0.), axis=-1)

    def fit_custom(self):
        print("Node:", self.device_number)
        print("Save path:", self.save_path)

        self.checkpoint.on_train_begin()
        self.csv_losses.on_train_begin()
        self.csv_valid_metrics.on_train_begin()
        self.csv_test_metrics.on_train_begin()

        # self.evaluate(0, "valid")
        # self.evaluate(0, "test")
        for i in range(1, self.epoch_num + 1):
            print("Epoch:", i)
            self.train(i)
            if i % self.epoch_num_valid == 0:
                self.evaluate(i, "valid")
                self.evaluate(i, "test")

        self.checkpoint.on_train_end()
        self.csv_losses.on_train_end()
        self.csv_test_metrics.on_train_end()
        self.csv_valid_metrics.on_train_end()

    def train(self, epoch):
        print("Train:")
        losses = []
        generator_train = self.reader.batch_generator_train()
        for el in generator_train:
            loss = self.obj_model.train_on_batch(x=el[0], y=el[1])
            losses.append(loss)
            print("loss:", loss)
        mean_loss = np.mean(np.asarray(losses).astype(float)) if len(losses) > 0 else -1.0

        print("Validation:")
        val_losses = []
        generator_valid = self.reader.batch_generator_train("valid")
        for el in generator_valid:
            val_loss = self.obj_model.test_on_batch(x=el[0], y=el[1])
            val_losses.append(val_loss)
            print("val_loss:", val_loss)
        mean_val_loss = np.mean(np.asarray(val_losses).astype(float)) if len(val_losses) > 0 else -1.0

        self.checkpoint.on_epoch_end(epoch, {"loss": mean_loss, "val_loss": mean_val_loss})
        self.csv_losses.on_epoch_end(epoch, {"loss": mean_loss, "val_loss": mean_val_loss})

    def evaluate(self, epoch, eval_type="valid"):
        if eval_type == "valid":
            steps = self.reader.valid_steps
            num_samples = self.reader.num_ranking_samples_valid
            generator = self.reader.batch_generator_test("valid")
        elif eval_type == "test":
            steps = self.reader.test_steps
            num_samples = self.reader.num_ranking_samples_test
            generator = self.reader.batch_generator_test("test")

        metrics_logs = {}
        y_true = []
        y_pred = []
        for el in generator:
            y_true.append(np.expand_dims(el[1], axis=1))
            y_pred.append(self.score_model.predict_on_batch(x=el[0]))

        y_true = np.vstack([np.hstack(y_true[i * num_samples:
                           (i + 1) * num_samples]) for i in range(steps)])
        y_pred = np.vstack([np.hstack(y_pred[i * num_samples:
                           (i + 1) * num_samples]) for i in range(steps)])

        for i in range(len(self.metrics)):
            metric_name = self.metrics[i]
            metric_value = self.metrics_functions[i](y_true, y_pred)
            metrics_logs[metric_name] = metric_value
            print(metric_name + ':', metric_value)

        if eval_type == "valid":
            self.csv_valid_metrics.on_epoch_end(epoch, metrics_logs)
        elif eval_type == "test":
            self.csv_test_metrics.on_epoch_end(epoch, metrics_logs)