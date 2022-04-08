# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Evaluation for FasterRcnn"""
import os
import argparse
import time
import numpy as np
from pycocotools.coco import COCO
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed, Parameter

from src.FasterRcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50
from src.config import config
from src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset
from src.util import coco_eval, bbox2result_1image, results2json

set_seed(1)

parser = argparse.ArgumentParser(description="FasterRcnn evaluation")
parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
parser.add_argument("--ann_file", type=str, default="val.json", help="Ann file, default is val.json.")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--device_target", type=str, default="Ascend",
                    help="device where the code will be implemented, default is Ascend")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)


def fasterrcnn_eval(dataset_path, ckpt_path, ann_file):
    """FasterRcnn evaluation."""

    # 1.build dataset (functions may be used: create_fasterrcnn_dataset)
    fasterrcnn_dataset = create_fasterrcnn_dataset(dataset_path, batch_size=config.test_batch_size, is_training=False)
    len_dataset = fasterrcnn_dataset.get_dataset_size()
    print('len of dataset: ', len_dataset)

    # 2.build network
    net = Faster_Rcnn_Resnet50(config=config)

    # 3.load trained checkpoint (functions may be used: load_checkpoint, load_param_into_net)
    ckpt = load_checkpoint(ckpt_path)
    load_param_into_net(net, ckpt)

    # 4.set eval mode
    net.set_train(False)

    # 5.set model as float16 for Ascend inference
    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

    # 6.inference process (functions may be used: bbox2result_1image)
    # Note: the inference process includes both model inference and post-process
    outputs = []
    max_num = 128  # max num of bboxes reserved for one image

    # inference
    for data in fasterrcnn_dataset.create_dict_iterator(num_epochs=1):
        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']

        output = net(img_data, img_metas, gt_bboxes, gt_labels, gt_num)
        all_bbox = output[0]
        all_label = output[1]
        all_mask = output[2]
        # print('------')
        # print('all_bbox_shape:', all_bbox.shape)
        # print('all_label_shape:', all_label.shape)
        # print('all_mask_shape:', all_mask.shape)
        # print('all_mask:')
        # print(all_mask)
        # print('------')

        # post-process
        for i in range(config.test_batch_size):
            all_bbox_squeezed = np.squeeze(all_bbox.asnumpy()[i, :, :])
            all_label_squeezed = np.squeeze(all_label.asnumpy()[i, :, :])
            all_mask_squeezed = np.squeeze(all_mask.asnumpy()[i, :, :])
            # print(i, ':')
            # print('all_bbox_squ:', all_bbox_squeezed)
            # print('all_label_squ:', all_label_squeezed)
            # print('all_mask_squ:', all_mask_squeezed)
            # print('--------')

            all_bboxes_mask = all_bbox_squeezed[all_mask_squeezed, :]
            all_labels_mask = all_label_squeezed[all_mask_squeezed]
            # print('all_bboxes_mask_num:', all_bboxes_mask.shape[0])
            # print('all_bboxes_mask:', all_bboxes_mask)
            # print('all_labels_mask:', all_labels_mask)
            #
            # if all_bboxes_mask.shape[0] > max_num:
            #     indexes = np.argsort(-all_bboxes_mask[:, -1])
            #     indexes = indexes[:max_num]
            #     all_bboxes_mask = all_bboxes_mask[indexes]
            #     all_labels_mask = all_labels_mask[indexes]
            #     print('all_bboxes_mask(after filtering):', all_bboxes_mask)

            outputs_tmp = bbox2result_1image(all_bboxes_mask, all_labels_mask, config.num_classes)
            outputs.append(outputs_tmp)

    # 7.load ground-truth annotations
    gt = COCO(ann_file)

    # 8.evaluation (functions may be used: results2json, coco_eval)
    result_files = results2json(gt, outputs, "./results.pkl")
    eval_types = ["bbox"]
    coco_eval(result_files, eval_types, gt)

if __name__ == '__main__':
    prefix = "FasterRcnn_eval.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    print("CHECKING MINDRECORD FILES ...")

    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if args_opt.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image("coco", False, prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        else:
            if os.path.isdir(config.IMAGE_DIR) and os.path.exists(config.ANNO_PATH):
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image("other", False, prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("IMAGE_DIR or ANNO_PATH not exits.")

    print("CHECKING MINDRECORD FILES DONE!")
    print("Start Eval!")
    fasterrcnn_eval(mindrecord_file, args_opt.checkpoint_path, args_opt.ann_file)
