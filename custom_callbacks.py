import numpy as np
import json
import custom_metrics
from keras.callbacks import Callback


class MetricsCallback(Callback):
    def __init__(self, data_reader, obj_model, score_model, score_model_by_name, params_dict):
        super(Callback, self).__init__()
        self.data_reader = data_reader
        self.obj_model = obj_model
        self.score_model = score_model
        self.score_model_by_name = score_model_by_name
        self.metrics = params_dict["metrics"]
        self.save_folder = params_dict["save_folder"]

    def on_train_begin(self, logs=None):
        self.metrics_functions = []
        self.val_metrics = {}
        self.test_metrics = {}
        for el in self.metrics:
            if el == "rank_resp":
                self.metrics_functions.append(custom_metrics.rank_response)
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
            if el == "rank_context":
                self.metrics_functions.append(custom_metrics.rank_context)
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

    def on_epoch_begin(self, epoch, logs=None):
        metrics_buff = self.evaluate("valid")
        for el in self.metrics:
            self.val_metrics[el].append(metrics_buff[el])
        print(self.val_metrics)
        metrics_buff = self.evaluate("test")
        for el in self.metrics:
            self.test_metrics[el].append(metrics_buff[el])
        print(self.test_metrics)
        with open(self.save_folder + '/valid_metrics.json', 'w') as outfile:
            json.dump(self.val_metrics, outfile)
        with open(self.save_folder + '/test_metrics.json', 'w') as outfile:
            json.dump(self.test_metrics, outfile)
        self.obj_model.save(self.save_folder + '/obj_model_' + str(epoch) + '.hdf5')
        self.score_model_by_name.save(self.save_folder + '/score_model_by_name_' + str(epoch) + '.hdf5')
        self.score_model.save(self.save_folder + '/score_model_' + str(epoch) + '.hdf5')

    def evaluate(self, eval_type="valid"):
        if eval_type == "valid":
            steps = self.data_reader.valid_steps
            num_samples = self.data_reader.num_ranking_samples_valid
            generator = self.data_reader.batch_generator_test("valid")
        elif eval_type == "test":
            steps = self.data_reader.test_steps
            num_samples = self.data_reader.num_ranking_samples_test
            generator = self.data_reader.batch_generator_test("test")

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
        for i in range(len(self.metrics)):
            metric_name = self.metrics[i]
            metric_value = self.metrics_functions[i](y_true, y_pred)
            metrics_buff[metric_name] = metric_value
            print(metric_name + ':', metric_value)

        return metrics_buff


class SaveCallback(Callback):
    def __init__(self, obj_model, params_dict):
        super(Callback, self).__init__()
        self.obj_model = obj_model
        self.save_folder = params_dict["save_folder"]

    def on_epoch_begin(self, epoch, logs=None):
        self.obj_model.save(self.save_folder + '/obj_model_' + str(epoch) + '.hdf5')

