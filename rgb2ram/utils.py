# come to save_datasets
import os
import random
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_datasets(game_name):
  # Copy images and rams from ../train_history/environments and save to "rgb_ram.npz"

  SOURCE_RGB_DIR = "../train_history/environments/{}-v4/rgb".format(game_name)
  SOURCE_RAM_DIR = "../train_history/environments/{}-v4/ram".format(game_name)

  assert(len(os.listdir(SOURCE_RGB_DIR)) == len(os.listdir(SOURCE_RAM_DIR)))

  rgb_files = []; rgb_filenames = []
  ram_files = []; ram_filenames = []
  for rgb_file, ram_file in zip(sorted(os.listdir(SOURCE_RGB_DIR)),
                                sorted(os.listdir(SOURCE_RAM_DIR))):
    #print(rgb_file, ram_file)
    assert(rgb_file[3:] == ram_file[3:])
    rgb_files.append(read_image(os.path.join(SOURCE_RGB_DIR, rgb_file)))
    ram_files.append(read_image(os.path.join(SOURCE_RAM_DIR, ram_file)))
    rgb_filenames.append(rgb_file)
    ram_filenames.append(ram_file)

  rgb_files = np.array(rgb_files)
  ram_files = np.array(ram_files)

  if not os.path.exists("data/{}-v4/".format(game_name)):
    os.makedirs("data/{}-v4/".format(game_name))

  np.savez("data/{}-v4/rgb_ram".format(game_name), rgb=rgb_files, ram=ram_files, rgb_names=rgb_filenames, ram_names=ram_filenames)

def load_data(game_name, model_type = None, split = 0.8, is_save_data = False):
  # Get train and val data and flatten them if FeedForward NN is the model
  x_train, y_train, x_test, y_test = load_datasets(is_save_data, split, game_name)
  input_dim = 84 * 84

  if model_type and "FFModel" in model_type.__name__:
    if x_train is not None: x_train = np.reshape(x_train, [-1, input_dim])
    if x_test is not None: x_test = np.reshape(x_test, [-1, input_dim])

  if y_train is not None: y_train = np.reshape(y_train, [-1, 128])
  if y_test is not None: y_test = np.reshape(y_test, [-1, 128])

  return x_train, y_train, x_test, y_test

def load_datasets(is_save_data, split, game_name):
  # load data from "rgb_ram.npz" and split into train and val

  random.seed(42)
  ldata = np.load("data/{}-v4/rgb_ram.npz".format(game_name))
  data = list(zip(ldata['rgb'], ldata['ram'], ldata['rgb_names'], ldata['ram_names']))

  random.shuffle(data)
  split_point = int(len(data) * split)

  train = data[:split_point]
  validation = data[split_point:]
  print("#train: %d, #validation: %d" % (len(train), len(validation)))

  train_rgb = train_ram = train_rgb_names = train_ram_names = None
  val_rgb = val_ram = val_rgb_names = val_ram_names = None

  if train: train_rgb, train_ram, train_rgb_names, train_ram_names = zip(*train)
  if validation: val_rgb, val_ram, val_rgb_names, val_ram_names = zip(*validation)

  if is_save_data: save_datasets(train_rgb, train_ram, val_rgb, val_ram,
    train_rgb_names, train_ram_names, val_rgb_names, val_ram_names)

  if train:
    train_rgb = np.array(train_rgb).astype('float32') / 255
    train_ram = np.array(train_ram).astype('float32') / 255

  if validation:
    val_rgb = np.array(val_rgb).astype('float32') / 255
    val_ram = np.array(val_ram).astype('float32') / 255

  return train_rgb, train_ram, val_rgb, val_ram

def save_datasets(train_rgb, train_ram, val_rgb, val_ram,
                  train_rgb_names, train_ram_names,
                  val_rgb_names, val_ram_names):
  # save train and val data in ./data within ram and rgb folders

  TRAIN_DIR = "./data/train"
  VAL_DIR = "./data/val"

  if not os.path.exists(TRAIN_DIR): os.makedirs(TRAIN_DIR)
  if not os.path.exists(VAL_DIR): os.makedirs(VAL_DIR)

  train_rgb_dir = os.path.join(TRAIN_DIR, "rgb/")
  train_ram_dir = os.path.join(TRAIN_DIR, 'ram/')
  val_rgb_dir = os.path.join(VAL_DIR, "rgb/")
  val_ram_dir = os.path.join(VAL_DIR, 'ram/')

  if os.path.exists(train_rgb_dir): shutil.rmtree(train_rgb_dir)
  if os.path.exists(train_ram_dir): shutil.rmtree(train_ram_dir)
  if os.path.exists(val_rgb_dir): shutil.rmtree(val_rgb_dir)
  if os.path.exists(val_ram_dir): shutil.rmtree(val_ram_dir)

  os.makedirs(train_rgb_dir)
  os.makedirs(train_ram_dir)
  os.makedirs(val_rgb_dir)
  os.makedirs(val_ram_dir)

  if train_rgb is not None:
    for train_rgb_e, train_ram_e, train_rgb_name, train_ram_name in zip(train_rgb, train_ram, train_rgb_names, train_ram_names):
      train_rgb_img = Image.fromarray(np.squeeze(train_rgb_e, axis = -1).astype(np.uint8))
      train_ram_img = Image.fromarray(np.squeeze(train_ram_e, axis = -1).astype(np.uint8))
      train_rgb_img.save(train_rgb_dir + train_rgb_name)
      train_ram_img.save(train_ram_dir + train_ram_name)
  if val_rgb is not None:
    for val_rgb_e, val_ram_e, val_rgb_name, val_ram_name in zip(val_rgb, val_ram, val_rgb_names, val_ram_names):
      val_rgb_img = Image.fromarray(np.squeeze(val_rgb_e, axis = -1).astype(np.uint8))
      val_ram_img = Image.fromarray(np.squeeze(val_ram_e, axis = -1).astype(np.uint8))
      val_rgb_img.save(val_rgb_dir + val_rgb_name)
      val_ram_img.save(val_ram_dir + val_ram_name)

"""
save_rgb_array(image, output_dir="./train_history/environments/rgb/",
               filename="_epi_{}_step_{}".format(episode, i)

def save_rgb_array(image_array, output_dir="", filename=""):
    img = Image.fromarray(image_array)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img.save(output_dir + '/rgb{}.png'.format(filename))
"""
"""
(Pdb) image_array.shape
(84, 84)
(Pdb) image_array.dtype
dtype('uint8')
(Pdb) type(image_array)
<class 'numpy.ndarray'>
"""

def read_image(image_path):
  """Get a numpy array of an image so that one can access values[x][y]."""
  image = Image.open(image_path, 'r')
  width, height = image.size
  pixel_values = list(image.getdata())
  if image.mode == 'RGB':
    channels = 3
  elif image.mode == 'L':
    channels = 1
  else:
    print("Unknown mode: %s" % image.mode)
    return None
  pixel_values = np.array(pixel_values).reshape((width, height, channels))
  return pixel_values

def plot_history(history, figure_name):
  if not os.path.exists('saved_figures/'):
      os.makedirs('saved_figures/')
  print(history.history.keys())
  # dict_keys(['val_loss', 'val_mse', 'val_mae', 'loss', 'mse', 'mae'])
  plt.plot(history.history['mse'])
  plt.plot(history.history['val_mse'])
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train MSE', 'val MSE'], loc='upper left')
  #plt.show()
  plt.savefig('saved_figures/{}_MSE.png'.format(figure_name))
  plt.plot(history.history['mae'], color = 'm')
  plt.plot(history.history['val_mae'], color = 'g')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train MAE', 'val MAE'], loc='upper left')
  #plt.show()
  plt.savefig('saved_figures/{}_MAE.png'.format(figure_name))
  print('Figures have been save to "saved_figures/"')


def save_model(model, model_type, model_path="./saved_model/"):
  # serialize model to JSON
  if not os.path.exists(model_path):
      os.makedirs(model_path)
  model_json = model.to_json()
  with open(os.path.join(model_path, model_type.__name__ + ".json"), "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(os.path.join(model_path, model_type.__name__ + ".h5"))
  print("Saved model to disk")

if __name__ == "__main__":
  get_datasets("Seaquest")
