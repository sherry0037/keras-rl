# Analysis of RAM data

from PIL import Image
import utils
import numpy as np
import matplotlib.pyplot as plt

rgbs, rams, a, b = utils.load_data(split = 1.0, is_save_data = True)

mean_rgb = np.mean(rgbs, axis = 0)
mean_ram = np.mean(rams, axis = 0) * 255

index = np.arange(len(mean_ram))

# mean_ram_img = Image.fromarray(np.expand_dims(mean_ram, axis=1).astype(np.uint8))
# mean_ram_img.save('/Users/nidhikadkol/Desktop/mean_ram_img.png')

plt.bar(index, mean_ram)
plt.ylabel('Normalized byte value', fontsize=10)
plt.xlabel('byte index', fontsize=10)
plt.title('Mean Ram')
plt.show()
