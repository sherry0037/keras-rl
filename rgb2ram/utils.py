import os
import random
import shutil
from PIL import Image
import numpy

random.seed(42)
TRAIN_DIR = "./data/train"
VAL_DIR = "./data/val"
SOURCE_RGB_DIR = "../train_history/environments/rgb"
SOURCE_RAM_DIR = "../train_history/environments/ram"
SPLIT_RATIO = [0.8, 0.2]


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

    train_rgb = numpy.stack(rgb_files, axis=0)
    train_ram = numpy.stack(ram_files, axis=0)
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

    val_rgb = numpy.stack(rgb_files, axis=0)
    val_ram = numpy.stack(ram_files, axis=0)
    print("Validation rgb matrix shape: {}, ram matrix shape: {}".format(val_rgb.shape, val_ram.shape))

    return train_rgb, train_ram, val_rgb, val_ram


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
    pixel_values = numpy.array(pixel_values).reshape((width, height, channels))
    return pixel_values

if __name__ == "__main__":
    get_datasets()
    #load_datasets()
    pass