import pandas as pd
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 1000)
plt.style.use('ggplot')


def get_all_file_names(path_folder):
    result_files = []
    for file in os.listdir(path_folder):
        if file.endswith('.hdf5'):
            result_files.append(path_folder + file[:len(file) - 4])
    return sorted(result_files)


def parse_values(dataset):
    ep_, loss_, val_ = [], [], []

    for chunk in dataset:
        chunk = chunk[18:len(chunk) - 1].split('_')
        ep_.append(int(chunk[0]))
        loss_.append(float(chunk[2]))
        val_.append(float(chunk[5]))

    params = pd.DataFrame()
    params['epoch'] = ep_
    params['loss'] = loss_
    params['val_loss'] = val_

    return params.sort_values(by='epoch')


files = get_all_file_names('models/')
data_params = parse_values(files)

print(data_params)

plt.plot(data_params['epoch'].values, data_params['loss'].values, lw=2, color='red')
plt.plot(data_params['epoch'].values, data_params['val_loss'].values, lw=2, color='green')
plt.title('Loss and validation loss')
plt.legend(('loss', 'val_loss'), loc='upper right')
plt.tight_layout()
plt.show()

