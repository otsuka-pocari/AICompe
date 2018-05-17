import pickle

from PIL import Image
from util import const


def array2img(np_array):

    np_array = np_array.reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(np_array)

    return img


def create_cifar10_dataset():

    for i in range(1, 6):
        with (const.CIFAR10_DIR / 'data_batch_{}'.format(i)).open('rb') as rf:
            data_batch = pickle.load(rf, encoding='bytes')
        images = [array2img(img_array) for img_array in data_batch[b'data']]

    with (const.CIFAR10_DIR / 'test_batch').open('rb') as rf:
        pass


def load_fasion-mnist_dataset():

    if not (const.TRAIN_DIR.exists() or const.TEST_DIR.exists()):
        const.TRAIN_DIR.mkdir(exist_ok=True)
        const.TEST_DIR.mkdir(exist_ok=True)
        create_cifar10_dataset()
