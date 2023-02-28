import numpy as np
import cv2
from typing import Union


def change_wh(xy):
    assert isinstance(xy, np.ndarray), f'[change_wh] value is not ndarray.'
    assert xy.shape[-1] == 4, f'[change_wh] shape is invalid. shape={xy.shape}'

    wh = np.zeros_like(xy, xy.dtype)

    wh[..., 0] = (xy[..., 2] + xy[..., 0]) * 0.5
    wh[..., 1] = (xy[..., 3] + xy[..., 1]) * 0.5
    wh[..., 2] = xy[..., 2] - xy[..., 0]
    wh[..., 3] = xy[..., 3] - xy[..., 1]

    return wh


def change_xy(wh):
    assert isinstance(wh, np.ndarray), f'[change_xy] value is not ndarray.'
    assert wh.shape[-1] == 4, f'[change_xy] shape is invalid. shape={wh.shape}'

    xy = np.zeros_like(wh, wh.dtype)

    xy[..., 0] = wh[..., 0] - wh[..., 2] * 0.5
    xy[..., 1] = wh[..., 1] - wh[..., 3] * 0.5
    xy[..., 2] = wh[..., 0] + wh[..., 2] * 0.5
    xy[..., 3] = wh[..., 1] + wh[..., 3] * 0.5

    return xy


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


def get_batch_bboxes(anchors, scales, y):
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

    bboxes = bboxes.reshape([-1, 4])
    scores = scores.reshape([-1, ])

    best_mark = scores > 0.6
    _, bboxes = nms(scores[best_mark], bboxes[best_mark], 0.3)

    return bboxes


def get_bboxes(anchors, scales, y):
    '''
    y: [nLayer, [1, h, w, 3, 4 + 1]]
    '''

    nLayer = len(y)
    t_elem = 0

    for y_output in y:
        h, w, a = y_output.shape[1:-1]
        t_elem += h * w * a

    scores = np.zeros([t_elem, ], np.float32)
    bboxes = np.zeros([t_elem, 4], np.int32)

    start_index = 0
    end_index = 0

    for j in range(nLayer):
        scale = scales[j]
        y_pred = np.squeeze(y[j], 0)  # [h, w, 3, 4 + 1]
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
        bboxes[start_index:end_index] = np.concatenate([x1y1, x2y2], -1).reshape([-1, 4])  # [h * w * 3, 4]
        scores[start_index:end_index] = y_pred[..., 4].reshape([-1, ])  # [h * w * 3]
        start_index = end_index

    best_mark = scores > 0.6
    _, bbox = nms(scores[best_mark], bboxes[best_mark], 0.3)

    return bbox


def resize_image(image, input_size):
    img_h, img_w = image.shape[:2]
    out_h, out_w = input_size

    canvus = np.zeros([out_h, out_w, 3], np.uint8)

    if img_h > img_w:
        ratio = out_h / img_h
        new_w = np.round(ratio * img_w).astype(np.int32)
        image = cv2.resize(image, [new_w, out_h])
        start = (out_w - new_w) // 2
        canvus[:, start:start + new_w, :] = image
    else:
        ratio = out_w / img_w
        new_h = np.round(ratio * img_h).astype(np.int32)
        image = cv2.resize(image, [out_w, new_h])
        start = (out_h - new_h) // 2
        canvus[start:start + new_h, :, :] = image

    return canvus





























