import os
import random
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)
TRAIN_DIR = "./data/train"
VAL_DIR = "./data/val"
SOURCE_RGB_DIR = "../train_history/environments/rgb"
SOURCE_RAM_DIR = "../train_history/environments/ram"


def get_datasets():
    """
    Copy images and rams from ../train_history/environments to ./data/
    :return:
    """
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    if not os.path.exists(VAL_DIR):
        os.makedirs(VAL_DIR)
    if not os.path.exists(SOURCE_RGB_DIR) or not os.path.exists(SOURCE_RAM_DIR):
        raise Exception('Cannot find source images.')

    rgb_files = []
    for file in os.listdir(SOURCE_RGB_DIR):
        if file.endswith(".png"):
            rgb_files.append(file)
    ram_files = []
    for file in os.listdir(SOURCE_RAM_DIR):
        if file.endswith(".png"):
            ram_files.append(file)

    ratio = [0.90, 0.08, 0.02]

    assert(len(rgb_files) == len(ram_files))
    data = list(zip(rgb_files, ram_files))
    print("=" * 20)
    print("Total number of data: %d" % len(data))
    random.shuffle(data)
    split_point = int(len(data) * ratio[0])
    train = data[:split_point]
    validation = data[split_point:]
    print("#train: %d, #validation: %d" % (len(train), len(validation)))
    print("Start generating datasets...")

    rgb_dir = os.path.join(TRAIN_DIR, "rgb/")
    ram_dir = os.path.join(TRAIN_DIR, 'ram/')
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir)
    if not os.path.exists(ram_dir):
        os.makedirs(ram_dir)

    for rgb, ram in train:
        shutil.copyfile(os.path.join(SOURCE_RGB_DIR, rgb), os.path.join(rgb_dir, rgb))
        shutil.copyfile(os.path.join(SOURCE_RAM_DIR, ram), os.path.join(ram_dir, ram))

    rgb_dir = os.path.join(VAL_DIR, "rgb/")
    ram_dir = os.path.join(VAL_DIR, 'ram/')
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir)
    if not os.path.exists(ram_dir):
        os.makedirs(ram_dir)

    for rgb, ram in validation:
        shutil.copyfile(os.path.join(SOURCE_RGB_DIR, rgb), os.path.join(rgb_dir, rgb))
        shutil.copyfile(os.path.join(SOURCE_RAM_DIR, ram), os.path.join(ram_dir, ram))


def load_datasets():
    # Load train data
    rgb_files = []
    for file in os.listdir(os.path.join(TRAIN_DIR, "rgb")):
        if file.endswith(".png"):
            rgb_files.append(read_image(os.path.join(TRAIN_DIR, "rgb", file)))
    ram_files = []
    for file in os.listdir(os.path.join(TRAIN_DIR, "ram")):
        if file.endswith(".png"):
            y = read_image(os.path.join(TRAIN_DIR, "ram", file)).reshape(128)
            ram_files.append(y)

    train_rgb = np.stack(rgb_files, axis=0)
    train_ram = np.stack(ram_files, axis=0)
    print("Train rgb matrix shape: {}, ram matrix shape: {}".format(train_rgb.shape, train_ram.shape))

    # Load validation data
    rgb_files = []
    for file in os.listdir(os.path.join(VAL_DIR, "rgb")):
        if file.endswith(".png"):
            rgb_files.append(read_image(os.path.join(VAL_DIR, "rgb", file)))
    ram_files = []
    for file in os.listdir(os.path.join(VAL_DIR, "ram")):
        if file.endswith(".png"):
            y = read_image(os.path.join(VAL_DIR, "ram", file)).reshape(128)
            ram_files.append(y)
    import pdb
    pdb.set_trace()
    val_rgb = np.stack(rgb_files, axis=0)
    val_ram = np.stack(ram_files, axis=0)
    print("Validation rgb matrix shape: {}, ram matrix shape: {}".format(val_rgb.shape, val_ram.shape))

    train_rgb = train_rgb.astype('float32') / 255
    train_ram = train_ram.astype('float32') / 255
    val_rgb = val_rgb.astype('float32') / 255
    val_ram = val_ram.astype('float32') / 255

    return train_rgb, train_ram, val_rgb, val_ram

def load_data(model_type=None):
    x_train, y_train, x_test, y_test = load_datasets()
    input_dim = x_train.shape[1] * x_train.shape[1]
    if model_type and "FFModel" in model_type.__name__:
        x_train = np.reshape(x_train, [-1, input_dim])
        x_test = np.reshape(x_test, [-1, input_dim])
    return x_train, y_train, x_test, y_test

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

if __name__ == "__main__":
    #get_datasets()
    load_datasets()
    pass

def plot_history(history):
  print(history.history.keys())
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()

def save_model(model):
  # serialize model to JSON
  model_path = "./saved_model/"
  if not os.path.exists(model_path):
      os.makedirs(model_path)
  model_json = model.to_json()
  with open(model_path+"model.json", "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(model_path+"model.h5")
  print("Saved model to disk")
