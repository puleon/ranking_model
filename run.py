import json
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from ranking_model import RankingModel

def main():
    f = open('./config.json')
    params_dict = json.load(f)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.visible_device_list = params_dict['device_number']
    set_session(tf.Session(config=config))

    rm = RankingModel(params_dict)

    tf.Session(config=config).close()

if __name__ == "__main__":
    main()

