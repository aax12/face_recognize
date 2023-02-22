import json
import cv2
import numpy as np
import os
from typing import Union
from model.face_detecter import TinyFaceDetection
from tensorflow.keras.utils import Progbar
import tensorflow as tf


def IOU(
        box_a: Union[list, tuple, np.ndarray],
        box_b: Union[list, tuple, np.ndarray]) -> np.ndarray:

    if not isinstance(box_a, np.ndarray):
        box_a = np.array(box_a, dtype=np.float32)
    if not isinstance(box_b, np.ndarray):
        box_b = np.array(box_b, dtype=np.float32)

    shape_a = box_a.shape
    shape_b = box_b.shape

    assert shape_a[-1] == 4, f'[IOU] IOU can not support box_a shape {shape_a}'
    assert shape_b[-1] == 4, f'[IOU] IOU can not support box_b shape {shape_b}'

    box_a = box_a.reshape([-1, 4])
    box_b = box_b.reshape([-1, 4])

    box_a = np.expand_dims(box_a, axis=1)
    box_b = np.expand_dims(box_b, axis=0)

    xa = np.maximum(box_a[..., 0], box_b[..., 0])
    ya = np.maximum(box_a[..., 1], box_b[..., 1])
    xb = np.minimum(box_a[..., 2], box_b[..., 2])
    yb = np.minimum(box_a[..., 3], box_b[..., 3])

    inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    a_area = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
    b_area = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])

    iou = inter / (a_area + b_area - inter + 1e-5)

    return iou.reshape([*shape_a[:-1], *shape_b[:-1]])


def nms(scores, bboxes, thresh):
    if isinstance(scores, np.ndarray):
        scores = np.array(scores)

    assert len(scores.shape) == 1, '[nms] score dimension이 {}입니다.'.format(scores.shape)
    assert len(scores) == len(bboxes), '[nms] score와 bbox 개수가 맞지 않습니다.'

    sort_index = np.argsort(scores)[::-1]
    sort_bboxes = bboxes[sort_index]

    keep = np.zeros(len(scores), 'bool')
    bbox_index = np.arange(len(scores))

    while len(bbox_index) > 1:
        select_index = bbox_index[0]
        other_index = bbox_index[1:]

        keep[select_index] = True

        iou = IOU(sort_bboxes[select_index], sort_bboxes[other_index])
        mask = iou < thresh

        bbox_index = other_index[mask]

    return keep, sort_bboxes[keep]


def get_bboxes(anchors, scales, y):
    '''
    y: [nLayer, [batch, h, w, 3, 4 + 1]]
    '''

    nLayer = len(y)
    batch = y[0].shape[0]
    t_elem = 0

    for y_output in y:
        h, w, a = y_output.shape[1:-1]
        t_elem += h * w * a

    scores = np.zeros([batch, t_elem, ], np.float32)
    bboxes = np.zeros([batch, t_elem, 4], np.int32)
    valid_bboxes = []

    for i in range(batch):
        start_index = 0
        end_index = 0

        for j in range(nLayer):
            scale = scales[j]
            y_pred = y[j][i]  # [h, w, 3, 4 + 1]
            height, width, n_anchor = y_pred.shape[:-1]

            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            grid_x = np.expand_dims(grid_x, -1)
            grid_y = np.expand_dims(grid_y, -1)
            grid = np.expand_dims(np.concatenate([grid_x, grid_y], -1), 2)  # [h, w, 1, 2]

            xy = scale * (y_pred[..., :2] + grid)  # [h, w, 3, 2]
            wh = np.exp(y_pred[..., 2:4]) * anchors[j]  # [h, w, 3, 2]

            x1y1 = (xy - wh // 2).astype(np.int32)
            x2y2 = (xy + wh // 2).astype(np.int32)

            end_index += height * width * n_anchor
            bboxes[i, start_index:end_index] = np.concatenate([x1y1, x2y2], -1).reshape([-1, 4])  # [h * w * 3, 4]
            scores[i, start_index:end_index] = y_pred[..., 4].reshape([-1, ])  # [h * w * 3]
            start_index = end_index

        best_mark = scores > 0.6
        _, bbox = nms(scores[best_mark], bboxes[best_mark], 0.3)

        valid_bboxes.append(bbox)

    return valid_bboxes


def read_partition(path, mode):
    with open(path, 'rt') as f:
        text = f.readlines()

    mask = np.zeros(len(text), 'bool')
    n_mode = {'train': '0', 'valid': '1', 'test': '2'}[mode]

    for i, line in enumerate(text):
        _, part = line.split()

        if part == n_mode:
            mask[i] = True

    return mask


def read_identity(path, mask):
    with open(path, 'rt') as f:
        text = f.readlines()

    names = []
    ident = []
    for line in text:
        name, n = line.split()

        ident.append(int(n))
        names.append(name)

    ident = np.array(ident)[mask]
    indices = np.arange(len(text))[mask]

    groups = []
    for n in set(ident):
        groups.append(indices[ident == n].tolist())

    return names, groups


def padding(images, image_size):
    canvus = np.ones((len(images), *image_size), 'uint8')
    canvus_h, canvus_w = image_size[:2]

    for i, img in enumerate(images):
        img_h, img_w = img.shape[:2]
        x1 = int((canvus_w - img_w) * 0.5)
        y1 = int((canvus_h - img_h) * 0.5)

        canvus[i, y1:y1+img_h, x1:x1+img_w] = img

    return canvus


def valid_bbox(bboxes):
    val_bbox = []

    for bbox in bboxes:
        if len(bbox) > 0:
            x1 = bbox[..., 0]
            y1 = bbox[..., 1]
            x2 = bbox[..., 2]
            y2 = bbox[..., 3]

            area = (x2 - x1) * (y2 * y1)
            max_index = np.argmax(area)
            val_bbox.append(bbox[max_index])
        else:
            val_bbox.append([])

    return val_bbox


def draw_rect(image, bbox):
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255))


def encoding_image(image, bbox, extend, resize):
    x1, y1, x2, y2 = bbox
    x1 -= extend
    y1 -= extend
    x2 += extend
    y2 += extend

    face_img = image[y1:y2, x1:x2]
    face_img = cv2.resize(face_img, resize)

    return cv2.imencode('.jpg', face_img)[1].tobytes()


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    if not isinstance(values, list):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def main():
    mode = 'train'
    amounts = 65536
    batch = 64

    with open('yolov3_tiny_cfg.json', 'r') as fp:
        config = json.load(fp)

    anchors = np.array(config['anchors'][::-1])
    scales = np.array(config['anchor_scale'][::-1])

    model = TinyFaceDetection(config['input_size'])
    model.load_weights('weights/pre_trained/face_detect_2stage.h5')

    eval_path = 'E:/data_set/CelebA/Eval/list_eval_partition.txt'
    identity_path = 'E:/data_set/CelebA/Anno/identity_CelebA.txt'
    image_dir = 'E:/data_set/CelebA/img/img_align_celeba'
    rec_path = 'E:/data_set/CelebA/train_224.record'

    mask = read_partition(eval_path, mode)
    names, groups = read_identity(identity_path, mask)

    print(f'groups {len(groups)}')

    i = 0
    bar = Progbar(amounts)
    option = tf.io.TFRecordOptions(compression_type='GZIP')
    with tf.io.TFRecordWriter(rec_path, option) as fp:
        while i < amounts:
            ap_group_indices = np.random.randint(0, len(groups), batch if i + batch < amounts else amounts - i)
            mask = np.ones(len(groups), 'bool')
            group_indices = np.arange(len(groups))

            anchor_img = []
            pos_img = []
            neg_img = []

            for ap_group_index in ap_group_indices:
                ap_group = groups[ap_group_index]

                if len(ap_group) > 1:
                    anchor_index, positive_index = np.random.choice(ap_group, 2, False)

                    mask[ap_group_index] = False
                    neg_group_index = np.random.choice(group_indices[mask])
                    neg_index = np.random.choice(groups[neg_group_index])

                    anchor_name = names[anchor_index]
                    pos_name = names[positive_index]
                    neg_name = names[neg_index]

                    anchor_img.append(cv2.imread(os.path.join(image_dir, anchor_name)))
                    pos_img.append(cv2.imread(os.path.join(image_dir, pos_name)))
                    neg_img.append(cv2.imread(os.path.join(image_dir, neg_name)))

            anchor_img = padding(anchor_img, config['input_size'])
            pos_img = padding(pos_img, config['input_size'])
            neg_img = padding(neg_img, config['input_size'])

            y = model.predict_on_batch(anchor_img)
            anchor_bboxes = valid_bbox(get_bboxes(anchors, scales, y))

            y = model.predict_on_batch(pos_img)
            pos_bboxes = valid_bbox(get_bboxes(anchors, scales, y))

            y = model.predict_on_batch(neg_img)
            neg_bboxes = valid_bbox(get_bboxes(anchors, scales, y))

            for j, (a, p, n) in enumerate(zip(anchor_bboxes, pos_bboxes, neg_bboxes)):
                if len(a) == len(p) == len(n) == 4:
                    encoded_anchor = encoding_image(anchor_img[j], a, 10, (224, 224))
                    encoded_pos = encoding_image(pos_img[j], p, 10, (224, 224))
                    encoded_neg = encoding_image(neg_img[j], n, 10, (224, 224))

                    feature = {
                        'image/anchor': bytes_feature(encoded_anchor),
                        'image/positive': bytes_feature(encoded_pos),
                        'image/negative': bytes_feature(encoded_neg),
                        'image/width': int64_feature(224),
                        'image/height': int64_feature(224)
                    }

                    fp.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
                    i += 1
                    bar.update(i)


def generator(image_dir, names, batch, input_size):
    count = 0

    while count < len(names):
        images = []
        file_names = []

        n = batch if (count + batch) < len(names) else len(names) - count
        for i in range(n):
            image = cv2.imread(os.path.join(image_dir, names[i + count]))
            images.append(image)
            file_names.append(names[i + count])

        count += n

        images = padding(images, input_size)

        yield images, file_names


def main2():
    mode = 'test'
    batch = 128

    with open('yolov3_tiny_cfg.json', 'r') as fp:
        config = json.load(fp)

    anchors = np.array(config['anchors'][::-1])
    scales = np.array(config['anchor_scale'][::-1])

    model = TinyFaceDetection(config['input_size'])
    model.load_weights('weights/pre_trained/face_detect_2stage.h5')

    eval_path = 'E:/data_set/CelebA/Eval/list_eval_partition.txt'
    image_dir = 'E:/data_set/CelebA/img/img_align_celeba'
    write_img_dir = 'E:/data_set/CelebA/img/face_images/test_images'

    with open(eval_path, 'rt') as fp:
        lines = fp.readlines()

    names = []
    for line in lines:
        name, part = line.split()

        if part == {'train': '0', 'valid': '1', 'test': '2'}[mode]:
            names.append(name)

    print(f'amounts {len(names)}')

    bar = Progbar(len(names))
    image_gen = generator(image_dir, names, batch, config['input_size'])

    i = 0
    file_names = []
    for images, filename in image_gen:
        y = model.predict_on_batch(images)
        bboxes = valid_bbox(get_bboxes(anchors, scales, y))

        for j, (bbox) in enumerate(bboxes):
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                x1 -= 10
                y1 -= 10
                x2 += 10
                y2 += 10

                face_img = images[j, y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(write_img_dir, filename[j]), face_img)
                file_names.append(filename[j])
        i += len(filename)
        bar.update(i)

    with open('E:/data_set/CelebA/img/face_images/test_names.txt', 'wt') as fp:
        for name in file_names:
            fp.write(name + '\n')

    print(f'total_images: {len(file_names)}')


if __name__ == '__main__':
    pass




































