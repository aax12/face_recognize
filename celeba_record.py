import numpy as np
import cv2
import tensorflow as tf
import os
from tensorflow.keras.utils import Progbar
from sample_data.celeb_a import CelebAParser


class WriteCelebA:
    @staticmethod
    def _bytes_feature(values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

    @staticmethod
    def _int64_feature(values):
        if not isinstance(values, list):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def __init__(self, celeba_dir):
        self.celeb_dir = celeba_dir

    def write(self, image_size, rec_path, mode):
        image_dir = os.path.join(self.celeb_dir, 'img', 'face_images', mode + '_images')
        file_path = os.path.join(self.celeb_dir, 'img', 'face_images', mode + '_names.txt')
        identity_path = os.path.join(self.celeb_dir, 'Anno', 'identity_CelebA.txt')
        attribute_path = os.path.join(self.celeb_dir, 'Anno', 'list_attr_celeba.txt')

        print('=====================================')
        print('Read annotation text')
        print('\n')
        print(f'Read attr text: {attribute_path}')
        # attr = self._read_attr(attribute_path)
        # print(f'attributes are {len(attr)}amounts.')
        # print('done!!!')
        # print('\n')
        print(f'Read identity text: {identity_path}')
        ident, files = self._read_identity(identity_path)
        print(f'identities are {len(ident)}amounts.')
        print('done!!!')
        print('\n')

        with open(file_path, 'rt') as fp:
            lines = fp.readlines()

        select_indices = []
        for line in lines:
            line = line.rstrip('\n')
            select_indices.append(int(line.rstrip('.jpg')) - 1)
        select_indices = np.array(select_indices)

        # print(f'Write attribution: {attr_path}')

        select_id = ident[select_indices]
        groups = []
        for n in set(select_id):
            groups.append(select_indices[select_id == n])

        '''
        with open(attr_path, 'wt') as f:
            for group in groups:
                for index in group:
                    f.write(f'{index} ')
                f.write('\n')
        '''
        print(f'{mode} image amounts are {len(select_id)}')
        print(f'{mode} image groups are {len(groups)}')
        print('\n')

        print(f'Write record: {rec_path}\n')

        i = 0
        bar = Progbar(np.count_nonzero(select_id))
        img_h, img_w = image_size[:2]

        option = tf.io.TFRecordOptions(compression_type='GZIP')
        with tf.io.TFRecordWriter(rec_path, option) as fp:
            for k, indices in enumerate(groups):
                faces = len(indices)
                wh_grid = np.ceil(np.sqrt(faces)).astype('int32')
                canvus = np.zeros((wh_grid * img_h, wh_grid * img_w, 3), 'uint8')

                images = []
                for index in indices:
                    images.append(cv2.imread(os.path.join(image_dir, files[index])))

                for n, image in enumerate(images):
                    x = n % wh_grid
                    y = n // wh_grid

                    canvus[y * img_h:(y + 1) * img_h, x * img_w:(x + 1) * img_w] = image

                    i += 1
                    bar.update(i)
    
                encoded_image = cv2.imencode('.jpg', canvus)[1].tobytes()
    
                feature = {
                    'image/source': WriteCelebA._bytes_feature(encoded_image),
                    'image/width': WriteCelebA._int64_feature(canvus.shape[1]),
                    'image/height': WriteCelebA._int64_feature(canvus.shape[0]),
                    'image/face_width': WriteCelebA._int64_feature(img_w),
                    'image/face_height': WriteCelebA._int64_feature(img_h),
                    'image/faces': WriteCelebA._int64_feature(faces),
                    'image/id': WriteCelebA._int64_feature(k)
                }

                fp.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
        print('\ndone!!!\n')

    def _read_identity(self, path):
        groups = []
        file_names = []

        with open(path, 'rt') as f:
            data = f.readlines()

        for line in data:
            name, identity = line.split()
            groups.append(int(identity))
            file_names.append(name)

        return np.array(groups), file_names

    def _read_attr(self, path):
        attr = []

        with open(path, 'rt') as f:
            data = f.readlines()

        for line in data[2:]:
            char = line.split()
            attr.append([1. if n == '1' else 0. for n in char[1:]])

        return np.array(attr)


def main():
    celeba = WriteCelebA('E:/data_set/CelebA')
    celeba.write((224, 224, 3), 'E:/data_set/CelebA/img/face_images/test_224.record', 'test')


def main2():
    pass


if __name__ == '__main__':
    main()










































































































