import numpy as np
import tensorflow as tf
import cv2


class CelebAParser:
    def __init__(self, record_file, batches):
        data_sets = tf.data.TFRecordDataset(record_file, compression_type='GZIP')
        data_sets = data_sets.map(CelebAParser.tf_record_extract)
        self.data_sets = iter(data_sets.shuffle(batches).batch(batches).repeat())

        self.sample_cnt = 0
        self.images, self.block_w, self.block_h, self.n_faces, self.face_ids = self._get_data()

    def _get_data(self):
        samples = next(self.data_sets)

        images = self._decode_image(samples)
        face_width = samples['face_width'].numpy()
        face_height = samples['face_height'].numpy()
        faces = samples['faces'].numpy()
        face_ids = samples['face_id'].numpy()

        return images, face_width, face_height, faces, face_ids

    @staticmethod
    def _decode_image(sample):
        raw_images = sample['source'].numpy()
        widths = sample['width'].numpy()
        heights = sample['height'].numpy()

        images = []

        for raw_image, width, height in zip(raw_images, widths, heights):
            image = cv2.imdecode(np.fromstring(raw_image, np.uint8), -1)
            images.append(np.reshape(image, (height, width, 3)))

        return images

    @staticmethod
    def get_image(images, face_index, block_size):
        block_w, block_h = block_size

        grid_w = images.shape[1] // block_w

        grid_x = face_index % grid_w
        grid_y = face_index // grid_w

        return images[grid_y * block_h:(grid_y + 1) * block_h, grid_x * block_w:(grid_x + 1) * block_w]

    def get_sampledata(self):
        if self.sample_cnt >= len(self.images):
            self.images, self.block_w, self.block_h, self.n_faces, self.face_ids = self._get_data()
            self.sample_cnt = 0

        image = self.images[self.sample_cnt]
        block_w = self.block_w[self.sample_cnt]
        block_h = self.block_h[self.sample_cnt]
        faces = self.n_faces[self.sample_cnt]
        face_id = self.face_ids[self.sample_cnt]

        self.sample_cnt += 1

        return image, (block_h, block_w), faces, face_id

    @staticmethod
    def tf_record_extract(serialized_data):
        key_to_features = {
            'image/source': tf.io.FixedLenFeature((), tf.string, ''),
            'image/width': tf.io.FixedLenFeature((), tf.int64, 0),
            'image/height': tf.io.FixedLenFeature((), tf.int64, 0),
            'image/face_width': tf.io.FixedLenFeature((), tf.int64, 0),
            'image/face_height': tf.io.FixedLenFeature((), tf.int64, 0),
            'image/faces': tf.io.FixedLenFeature((), tf.int64, 0),
            'image/id': tf.io.FixedLenFeature((), tf.int64, 0)
        }
        features = tf.io.parse_single_example(serialized=serialized_data, features=key_to_features)

        image = tf.cast(features['image/source'], tf.string)
        width = tf.cast(features['image/width'], tf.int64)
        height = tf.cast(features['image/height'], tf.int64)
        face_width = tf.cast(features['image/face_width'], tf.int64)
        face_height = tf.cast(features['image/face_height'], tf.int64)
        faces = tf.cast(features['image/faces'], tf.int64)
        face_id = tf.cast(features['image/id'], tf.int64)

        return {'source': image, 'width': width, 'height': height, 'face_width': face_width, 'face_height': face_height, 'faces': faces, 'face_id': face_id}


class PairwiseCelebA:
    def __init__(self, input_size, celeba: CelebAParser, attr_path, group_path, batches, n_groups):
        self.n_groups = n_groups
        self.batches = batches
        self.celeba = celeba
        self.input_size = input_size

        with open(group_path, 'rt') as f:
            lines = f.readlines()

        self.indices = []
        for line in lines:
            _indices = []
            for num in line.split():
                _indices.append(int(num))
            self.indices.append(_indices)

        with open(attr_path, 'rt') as f:
            lines = f.readlines()[2:]

        attr = []
        for line in lines:
            attr.append([1. if int(num) > 0 else 0. for num in line.split()[1:]])

        self.attr = np.array(attr)

    def __iter__(self):
        return self

    def __next__(self):
        face_images = []
        face_labels = []
        face_attrs = []

        selected_ids = set()

        batch_ratio = self.batches / self.n_groups

        for i in range(self.n_groups):
            amounts = (np.round(batch_ratio * (i + 1)) - np.round(batch_ratio * i)).astype('int32')

            while True:
                image, block_size, faces, nid = self.celeba.get_sampledata()

                if nid in selected_ids or faces < amounts:
                    continue

                indices = np.random.choice(faces, amounts, False)

                for index in indices:
                    face_images.append(self.celeba.get_image(image, index, self.input_size))
                    face_attrs.append(self.attr[self.indices[nid][index]])

                face_labels.extend([nid] * amounts)

                selected_ids.add(nid)
                break

        return np.stack(face_images), (np.array(face_attrs), np.expand_dims(face_labels, -1))


class ContrastCelebA:
    def __init__(self, parser, batch, n_seq):
        self.parser = parser
        self.batch = batch
        self.n_seq = n_seq

    def __iter__(self):
        return self

    def __next__(self):
        x1_img = []
        x2_img = []
        truth = []

        for i in range(self.batch):
            x1 = []
            x2 = []
            while True:
                image, block_size, faces, face_id = self.parser.get_sampledata()

                if i % 2 == 0:
                    if faces >= self.n_seq * 2:
                        indices = np.random.choice(faces, self.n_seq * 2, False)

                        for index in indices[:self.n_seq]:
                            x1.append(cv2.resize(self.parser.get_image(image, index, block_size), (160, 160)))
                        for index in indices[self.n_seq:]:
                            x2.append(cv2.resize(self.parser.get_image(image, index, block_size), (160, 160)))

                        truth.append(1)
                    else:
                        continue
                else:
                    if faces >= self.n_seq:
                        indices = np.random.choice(faces, self.n_seq, False)

                        for index in indices[:self.n_seq]:
                            x1.append(cv2.resize(self.parser.get_image(image, index, block_size), (160, 160)))
                    else:
                        continue

                    while True:
                        n_image, n_block_size, n_faces, n_face_id = self.parser.get_sampledata()

                        if face_id != n_face_id and n_faces >= self.n_seq:
                            for index in np.random.choice(n_faces, self.n_seq, False):
                                x2.append(cv2.resize(self.parser.get_image(n_image, index, n_block_size), (160, 160)))
                            truth.append(0)
                            break
                        else:
                            continue
                break

            x1_img.append(x1)
            x2_img.append(x2)

        return (np.array(x1_img), np.array(x2_img)), np.array(truth)


class MultipleCelebA:
    def __init__(self, celeba: CelebAParser, n_groups, group_amounts, n_seq):
        self.n_groups = n_groups
        self.group_amounts = group_amounts
        self.celeba = celeba
        self.n_seq = n_seq

    def __iter__(self):
        return self

    def __next__(self):
        face_images = []
        face_labels = []

        selected_ids = set()

        for i in range(self.n_groups):
            while True:
                image, block_size, faces, nid = self.celeba.get_sampledata()

                if nid in selected_ids or faces < self.n_seq:
                    continue

                for _ in range(self.group_amounts):
                    group_faces = []
                    for index in np.random.choice(faces, self.n_seq, False):
                        group_faces.append(cv2.resize(self.celeba.get_image(image, index, block_size), [160, 160]))
                    face_images.append(group_faces)

                face_labels.extend([i] * self.group_amounts)

                selected_ids.add(nid)
                break

        return np.array(face_images), np.expand_dims(np.array(face_labels), -1)






















