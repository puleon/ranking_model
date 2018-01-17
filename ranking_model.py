from keras.layers import Input, LSTM, Dropout, Lambda, Dense, Activation
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.optimizers import SGD, Adam
from keras.initializers import glorot_uniform, Orthogonal
from keras import backend as K
from keras import losses
from keras import callbacks
from keras.models import load_model
import os
import numpy as np
from data_reader import DataReader
import custom_callbacks
import custom_layers
import custom_metrics
import json

class RankingModel(object):

    def __init__(self, params_dict):
        self.run_type = params_dict.get("run_type")
        self.model_file = params_dict.get("model_file")
        self.pooling = params_dict.get("pooling")
        self.save_folder = params_dict['save_folder']
        self.max_sequence_length = params_dict['max_sequence_length']
        self.embedding_dim = params_dict['embedding_dim']
        self.seed = params_dict['seed']
        self.hidden_dim = params_dict['hidden_dim']
        self.recdrop_val = params_dict['recdrop_val']
        self.inpdrop_val = params_dict['inpdrop_val']
        self.ldrop_val = params_dict['ldrop_val']
        self.dropout_val = params_dict['dropout_val']
        self.learning_rate = params_dict['learning_rate']
        self.margin = params_dict['margin']
        self.type_of_weights = params_dict['type_of_weights']
        self.epoch_num = params_dict["epoch_num"]
        self.metrics = params_dict["metrics"]

        if params_dict["type_of_loss"] == "triplet_hinge":
            self.loss = self.triplet_loss
            self.obj_model = self.triplet_hinge_loss_model()
        elif params_dict["type_of_loss"] == "binary_crossentropy":
            self.loss = losses.binary_crossentropy
            self.obj_model = self.binary_crossentropy_model()

        if params_dict['optimizer_name'] == 'SGD':
            self.optimizer = SGD(lr=self.learning_rate)
        elif params_dict['optimizer_name'] == 'Adam':
                self.optimizer = Adam(lr=self.learning_rate)

        if self.run_type == "train" or self.run_type is None:
            self.compile()
            self.score_model_by_name = self.obj_model.get_layer(name="score_model")
            self.reader = DataReader(self.score_model, params_dict)
            self.callbacks = []
            cb = custom_callbacks.MetricsCallback(self.reader, self.obj_model,
                                                  self.score_model, self.score_model_by_name, params_dict)
            self.callbacks.append(cb)
            for el in params_dict["callbacks"]:
                if el == "ModelCheckpoint":
                    folder_name = self.save_folder + "/weights"
                    os.mkdir(folder_name)
                    mc = callbacks.ModelCheckpoint(filepath=folder_name + "/weights.{epoch:02d}-{loss:.2f}.hdf5",
                                          monitor="val_loss",
                                          save_best_only=True,
                                          mode="min",
                                          verbose=1)
                    self.callbacks.append(mc)
            #     if el == "EarlyStopping":
            #         es = callbacks.EarlyStopping(monitor="val_loss", patience=19, mode="min")
            #         self.callbacks.append(es)
                if el == "TensorBoard":
                    folder_name = self.save_folder + "/logs"
                    tb = callbacks.TensorBoard(log_dir=folder_name)
                    self.callbacks.append(tb)
            #     if el == "ReduceLROnPlateau":
            #         rop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, mode="max")
            #         self.callbacks.append(rop)
                if el == "CSVLogger":

                    scvl = callbacks.CSVLogger(self.save_folder + '/training.log')
                    self.callbacks.append(scvl)

            self.fit_custom()

        elif self.run_type == "infer":
            self.compile()
            self.load()
            self.score_model = self.obj_model.get_layer(name="score_model")
            self.score_model_by_name = self.obj_model.get_layer(name="score_model")
            self.reader = DataReader(self.score_model, params_dict)
            # cb = custom_callbacks.MetricsCallback(self.reader, self.obj_model,
            #                                       self.score_model, self.score_model_by_name,
            #                                       params_dict)
            #cb.on_train_begin()
            #cb.on_epoch_begin(1)
            self.init_metrics()
            self.evaluate("valid")
            self.evaluate("test")
            self.save_metrics()

    def cosine(self, inputs):
        """Define a function for a lambda layer of a model."""

        input_a, input_b = inputs
        input_a = Lambda(lambda x: K.l2_normalize(x, -1))(input_a)
        input_b = Lambda(lambda x: K.l2_normalize(x, -1))(input_b)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=1))([input_a, input_b])
        return output

    def cosine_output_shape(self, shapes):
        """Define an output shape of a lambda layer of a model."""

        shape_a, shape_b = shapes
        return shape_a[0], 1

    def score_difference(self, inputs):
        """Define a function for a lambda layer of a model."""

        return Lambda(lambda x: x[0] - x[1])(inputs)

    def score_difference_output_shape(self, shapes):
        """Define an output shape of a lambda layer of a model."""

        return shapes[0]

    def sigmoid(self, inputs):
        input_a, input_b = inputs
        input_a = Dense(units=2 * self.max_sequence_length, use_bias=False)(input_a)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=1))([input_a, input_b])
        #output = custom_layers.Bias()(output)
        output = Activation("sigmoid")(output)
        return output

    def sigmoid_output_shape(self, shapes):
        """Define an output shape of a lambda layer of a model."""

        shape_a, shape_b = shapes
        return shape_a[0], 1

    def max_pooling(self, input):
        """Define a function for a lambda layer of a model."""

        return K.max(input, axis=1, keepdims=False)

    def max_pooling_output_shape(self, shape):
        """Define an output shape of a lambda layer of a model."""

        return shape[0], shape[2]

    def create_lstm_layer_max_pooling(self, input_dim):
        """Create a LSTM layer of a model."""

        inp = Input(shape=(input_dim,  self.embedding_dim,))
        inp_drop = Dropout(self.ldrop_val)(inp)
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        bioutp = Bidirectional(LSTM(self.hidden_dim,
                                    input_shape=(input_dim, self.embedding_dim,),
                                    kernel_regularizer=None,
                                    recurrent_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    recurrent_dropout=self.recdrop_val,
                                    dropout=self.inpdrop_val,
                                    kernel_initializer=ker_in,
                                    recurrent_initializer=rec_in,
                                    return_sequences=True), merge_mode='concat')(inp_drop)
        bioutp = Dropout(self.dropout_val)(bioutp)
        model = Model(inputs=inp, outputs=bioutp)
        return model

    def create_lstm_layer(self, input_dim):
        """Create a LSTM layer of a model."""

        inp = Input(shape=(input_dim,  self.embedding_dim,))
        inp_drop = Dropout(self.ldrop_val)(inp)
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        bioutp = Bidirectional(LSTM(self.hidden_dim,
                                    input_shape=(input_dim, self.embedding_dim,),
                                    kernel_regularizer=None,
                                    recurrent_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    recurrent_dropout=self.recdrop_val,
                                    dropout=self.inpdrop_val,
                                    kernel_initializer=ker_in,
                                    recurrent_initializer=rec_in,
                                    return_sequences=False), merge_mode='concat')(inp_drop)
        bioutp = Dropout(self.dropout_val)(bioutp)
        model = Model(inputs=inp, outputs=bioutp)
        return model

    def maxpool_cosine_score_model_separate_weights(self, input_dim):
        """Define a model with bi-LSTM layers and without attention."""

        input_a = Input(shape=(input_dim, self.embedding_dim,))
        input_b = Input(shape=(input_dim, self.embedding_dim,))
        lstm_layer_a = self.create_lstm_layer_max_pooling(self.max_sequence_length)
        lstm_layer_b = self.create_lstm_layer_max_pooling(self.max_sequence_length)
        lstm_last_a = lstm_layer_a(input_a)
        lstm_last_b = lstm_layer_b(input_b)
        max_pool_a = Lambda(self.max_pooling, output_shape=self.max_pooling_output_shape,
                      name="max_pooling_a")(lstm_last_a)
        max_pool_b = Lambda(self.max_pooling, output_shape=self.max_pooling_output_shape,
                      name="max_pooling_b")(lstm_last_b)
        cosine = Lambda(self.cosine, output_shape=self.cosine_output_shape,
                      name="similarity_network")([max_pool_a, max_pool_b])
        model = Model([input_a, input_b], cosine, name="score_model")
        return model

    def maxpool_cosine_score_model_shared_weights(self, input_dim):
        """Define a model with bi-LSTM layers and without attention."""

        input_a = Input(shape=(input_dim, self.embedding_dim,))
        input_b = Input(shape=(input_dim, self.embedding_dim,))
        lstm_layer = self.create_lstm_layer_max_pooling(self.max_sequence_length)
        lstm_last_a = lstm_layer(input_a)
        lstm_last_b = lstm_layer(input_b)
        max_pool_a = Lambda(self.max_pooling, output_shape=self.max_pooling_output_shape,
                      name="max_pooling_a")(lstm_last_a)
        max_pool_b = Lambda(self.max_pooling, output_shape=self.max_pooling_output_shape,
                      name="max_pooling_b")(lstm_last_b)
        cosine = Lambda(self.cosine, output_shape=self.cosine_output_shape,
                      name="similarity_network")([max_pool_a, max_pool_b])
        model = Model([input_a, input_b], cosine, name="score_model")
        return model

    def cosine_score_model_separate_weights(self, input_dim):
        """Define a model with bi-LSTM layers and without attention."""

        input_a = Input(shape=(input_dim, self.embedding_dim,))
        input_b = Input(shape=(input_dim, self.embedding_dim,))
        lstm_layer_a = self.create_lstm_layer(self.max_sequence_length)
        lstm_layer_b = self.create_lstm_layer(self.max_sequence_length)
        lstm_last_a = lstm_layer_a(input_a)
        lstm_last_b = lstm_layer_b(input_b)
        cosine = Lambda(self.cosine, output_shape=self.cosine_output_shape,
                      name="similarity_network")([lstm_last_a, lstm_last_b])
        model = Model([input_a, input_b], cosine, name="score_model")
        return model

    def cosine_score_model_shared_weights(self, input_dim):
        """Define a model with bi-LSTM layers and without attention."""

        input_a = Input(shape=(input_dim, self.embedding_dim,))
        input_b = Input(shape=(input_dim, self.embedding_dim,))
        lstm_layer = self.create_lstm_layer(self.max_sequence_length)
        lstm_last_a = lstm_layer(input_a)
        lstm_last_b = lstm_layer(input_b)
        cosine = Lambda(self.cosine, output_shape=self.cosine_output_shape,
                      name="similarity_network")([lstm_last_a, lstm_last_b])
        model = Model([input_a, input_b], cosine, name="score_model")
        return model

    def triplet_hinge_loss_model(self):
        question = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        answer_positive = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        answer_negative = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        if self.pooling is None or self.pooling == "max":
            if self.type_of_weights == "shared":
                self.score_model = self.maxpool_cosine_score_model_shared_weights(self.max_sequence_length)
            elif self.type_of_weights == "separate":
                    self.score_model = self.maxpool_cosine_score_model_separate_weights(self.max_sequence_length)
        elif self.pooling == "no":
            if self.type_of_weights == "shared":
                self.score_model = self.cosine_score_model_shared_weights(self.max_sequence_length)
            elif self.type_of_weights == "separate":
                    self.score_model = self.cosine_score_model_separate_weights(self.max_sequence_length)
        score_positive = self.score_model([question, answer_positive])
        score_negative = self.score_model([question, answer_negative])
        score_diff = Lambda(self.score_difference, output_shape=self.score_difference_output_shape,
                      name="score_diff")([score_positive, score_negative])
        model = Model([question, answer_positive, answer_negative], score_diff)
        return model

    def triplet_loss(self, y_true, y_pred):
        """Triplet loss function"""

        return K.mean(K.maximum(self.margin - y_pred, 0.), axis=-1)

    def maxpool_sigmoid_score_model_separate_weights(self, input_dim):
        """Define a model with bi-LSTM layers and without attention."""

        input_a = Input(shape=(input_dim, self.embedding_dim,))
        input_b = Input(shape=(input_dim, self.embedding_dim,))
        lstm_layer_a = self.create_lstm_layer_max_pooling(self.max_sequence_length)
        lstm_layer_b = self.create_lstm_layer_max_pooling(self.max_sequence_length)
        lstm_last_a = lstm_layer_a(input_a)
        lstm_last_b = lstm_layer_b(input_b)
        max_pool_a = Lambda(self.max_pooling, output_shape=self.max_pooling_output_shape,
                      name="max_pooling_a")(lstm_last_a)
        max_pool_b = Lambda(self.max_pooling, output_shape=self.max_pooling_output_shape,
                      name="max_pooling_b")(lstm_last_b)
        sigmoid = Lambda(self.sigmoid, output_shape=self.sigmoid_output_shape,
                      name="similarity_network")([max_pool_a, max_pool_b])
        model = Model([input_a, input_b], sigmoid, name="score_model")
        return model

    def maxpool_sigmoid_score_model_shared_weights(self, input_dim):
        """Define a model with bi-LSTM layers and without attention."""

        input_a = Input(shape=(input_dim, self.embedding_dim,))
        input_b = Input(shape=(input_dim, self.embedding_dim,))
        lstm_layer = self.create_lstm_layer_max_pooling(self.max_sequence_length)
        lstm_last_a = lstm_layer(input_a)
        lstm_last_b = lstm_layer(input_b)
        max_pool_a = Lambda(self.max_pooling, output_shape=self.max_pooling_output_shape,
                      name="max_pooling_a")(lstm_last_a)
        max_pool_b = Lambda(self.max_pooling, output_shape=self.max_pooling_output_shape,
                      name="max_pooling_b")(lstm_last_b)
        sigmoid = Lambda(self.sigmoid, output_shape=self.sigmoid_output_shape,
                      name="similarity_network")([max_pool_a, max_pool_b])
        model = Model([input_a, input_b], sigmoid, name="score_model")
        return model

    def binary_crossentropy_model(self):
        question = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        answer = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        if self.type_of_weights == "shared":
            self.score_model = self.maxpool_sigmoid_score_model_shared_weights(self.max_sequence_length)
        else:
            if self.type_of_weights == "separate":
                self.score_model = self.maxpool_sigmoid_score_model_separate_weights(self.max_sequence_length)
        score = self.score_model([question, answer])
        model = Model([question, answer], score)
        return model

    def compile(self):
        self.obj_model.compile(loss=self.loss,
                               optimizer=self.optimizer)

    def fit(self):
        self.obj_model.fit_generator(generator=self.reader.batch_generator_train(),
                                     steps_per_epoch=self.reader.train_steps,
                                     epochs=self.epoch_num,
                                     validation_data=self.reader.batch_generator_train("valid"),
                                     validation_steps=self.reader.valid_steps_train,
                                     callbacks=self.callbacks)

    def fit_custom(self):
        self.init_metrics()
        self.evaluate("valid")
        self.evaluate("test")
        self.save_metrics()
        for i in range(self.epoch_num):
            self.train()
            self.evaluate("valid")
            self.evaluate("test")
            self.save_weights(i)
            self.save_metrics()

    def train(self):
        print("Train:")
        losses = []
        generator_train = self.reader.batch_generator_train()
        for el in generator_train:
            loss = self.obj_model.train_on_batch(x=el[0], y=el[1])
            losses.append(loss)
            print("loss:", loss)
        self.losses["loss"].append(np.mean(np.asarray(losses).astype(float)) if len(losses) > 0 else -1.0)

        print("Validation:")
        val_losses = []
        generator_valid = self.reader.batch_generator_train("valid")
        for el in generator_valid:
            val_loss = self.obj_model.test_on_batch(x=el[0], y=el[1])
            val_losses.append(val_loss)
            print("val_loss:", val_loss)
        self.losses["val_loss"].append(np.mean(np.asarray(val_losses).astype(float)) if len(val_losses) > 0 else -1.0)

    def evaluate(self, eval_type="valid"):
        if eval_type == "valid":
            steps = self.reader.valid_steps
            num_samples = self.reader.num_ranking_samples_valid
            generator = self.reader.batch_generator_test("valid")
        elif eval_type == "test":
            steps = self.reader.test_steps
            num_samples = self.reader.num_ranking_samples_test
            generator = self.reader.batch_generator_test("test")

        metrics_buff = {}
        y_true = []
        y_pred = []
        for el in generator:
            y_true.append(np.expand_dims(el[1], axis=1))
            y_pred.append(self.score_model.predict_on_batch(x=el[0]))

        y_true = np.vstack([np.hstack(y_true[i * num_samples:
                                                          (i + 1) * num_samples]) for i in range(steps)])
        y_pred = np.vstack([np.hstack(y_pred[i * num_samples:
                                                          (i + 1) * num_samples]) for i in range(steps)])
        print(np.flip(np.argsort(y_pred, -1), -1))
        for i in range(len(self.metrics)):
            metric_name = self.metrics[i]
            metric_value = self.metrics_functions[i](y_true, y_pred)
            metrics_buff[metric_name] = metric_value
            print(metric_name + ':', metric_value)

        if eval_type == "valid":
            for el in self.metrics:
                self.val_metrics[el].append(metrics_buff[el])
            print(self.val_metrics)
        elif eval_type == "test":
            for el in self.metrics:
                self.test_metrics[el].append(metrics_buff[el])
            print(self.test_metrics)

    def init_metrics(self):
        self.metrics_functions = []
        self.val_metrics = {}
        self.test_metrics = {}
        self.losses = {"loss": [], "val_loss": []}
        for el in self.metrics:
            if el == "rank_response":
                self.metrics_functions.append(custom_metrics.rank_response)
                self.val_metrics[el] = []
                self.test_metrics[el] = []
            if el == "rank_context":
                self.metrics_functions.append(custom_metrics.rank_context)
                self.val_metrics[el] = []
                self.test_metrics[el] = []
            if el == "r_at_1":
                self.metrics_functions.append(custom_metrics.r_at_1)
                self.val_metrics[el] = []
                self.test_metrics[el] = []
            if el == "r_at_2":
                self.metrics_functions.append(custom_metrics.r_at_2)
                self.val_metrics[el] = []
                self.test_metrics[el] = []
            if el == "r_at_5":
                self.metrics_functions.append(custom_metrics.r_at_5)
                self.val_metrics[el] = []
                self.test_metrics[el] = []
            if el == "map_at_1_full":
                self.metrics_functions.append(custom_metrics.map_at_1_full)
                self.val_metrics[el] = []
                self.test_metrics[el] = []
            if el == "map_at_1_relevant":
                self.metrics_functions.append(custom_metrics.map_at_1_relevant)
                self.val_metrics[el] = []
                self.test_metrics[el] = []
            if el == "diff_top":
                self.metrics_functions.append(custom_metrics.diff_top)
                self.val_metrics[el] = []
                self.test_metrics[el] = []
            if el == "diff_answer":
                self.metrics_functions.append(custom_metrics.diff_answer)
                self.val_metrics[el] = []
                self.test_metrics[el] = []

    def save_metrics(self):
        with open(self.save_folder + '/losses.json', 'w') as outfile:
            json.dump(self.losses, outfile)
        with open(self.save_folder + '/valid_metrics.json', 'w') as outfile:
            json.dump(self.val_metrics, outfile)
        with open(self.save_folder + '/test_metrics.json', 'w') as outfile:
            json.dump(self.test_metrics, outfile)

    def save_weights(self, epoch):
        self.obj_model.save_weights(self.save_folder + '/obj_model_' + str(epoch) + '.h5')
        self.score_model_by_name.save_weights(self.save_folder + '/score_model_by_name_' + str(epoch) + '.h5')
        self.score_model.save_weights(self.save_folder + '/score_model_' + str(epoch) + '.h5')

    def load(self):
        if self.model_file is not None and os.path.isfile(self.model_file):
            #self.obj_model = load_model(self.model_file)
            self.obj_model.load_weights(self.model_file)
        else:
            print("The model_file parameter is not set or is incorrect. Exit.")
            exit()