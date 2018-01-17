import json
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os
from shutil import copyfile
from ranking_model import RankingModel
import custom_callbacks

def main():
    f = open('./config.json')
    params_dict = json.load(f)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.visible_device_list = params_dict['device_number']
    set_session(tf.Session(config=config))

    save_folder = params_dict["save_folder"]
    if os.path.isdir(save_folder):
        print("The folder exists. Exit.")
        exit()
    else:
        os.mkdir(save_folder)
    copyfile('./config.json', save_folder + '/config.json')
    rm = RankingModel(params_dict)

    tf.Session(config=config).close()

if __name__ == "__main__":
    main()

