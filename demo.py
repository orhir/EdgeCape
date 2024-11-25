import argparse
import copy
import pickle
import random
import cv2
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint
from mmpose.core import wrap_fp16_model
from mmpose.models import build_posenet
from torchvision import transforms
from capeformer import *  # noqa
import torchvision.transforms.functional as F

from capeformer.models.utils.visualization import old_plot_results, plot_results

COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]


def shuffle_skeleton(skeleton):
    shuffle_skeleton = []
    max_limb = max([sublist[-1] for sublist in skeleton])
    values = np.arange(0, max_limb + 1)
    skeleton_set = set()
    while len(skeleton_set) < len(skeleton):
        skeleton_set.add(tuple(np.random.choice(values, size=2, replace=False)))
    return list(skeleton_set)


class Resize_Pad:
    def __init__(self, w=256, h=256):
        self.w = w
        self.h = h

    def __call__(self, image):
        _, w_1, h_1 = image.shape
        ratio_1 = w_1 / h_1
        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != 1:
            # padding to preserve aspect ratio
            if ratio_1 > 1:  # Make the image higher
                hp = int(w_1 - h_1)
                hp = hp // 2
                image = F.pad(image, (hp, 0, hp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w])
            else:
                wp = int(h_1 - w_1)
                wp = wp // 2
                image = F.pad(image, (0, wp, 0, wp), 0, "constant")
                return F.resize(image, [self.h, self.w])
        else:
            return F.resize(image, [self.h, self.w])


def transform_keypoints_to_pad_and_resize(keypoints, image_size):
    trans_keypoints = keypoints.clone()
    h, w = image_size[:2]
    ratio_1 = w / h
    if ratio_1 > 1:
        # width is bigger than height - pad height
        hp = int(w - h)
        hp = hp // 2
        trans_keypoints[:, 1] = keypoints[:, 1] + hp
        trans_keypoints *= (256. / w)
    else:
        # height is bigger than width - pad width
        wp = int(image_size[1] - image_size[0])
        wp = wp // 2
        trans_keypoints[:, 0] = keypoints[:, 0] + wp
        trans_keypoints *= (256. / h)
    return trans_keypoints


def parse_args():
    parser = argparse.ArgumentParser(description='Pose Anything Demo')
    parser.add_argument('--support', help='Image file')
    parser.add_argument('--query', help='Image file')
    parser.add_argument('--config', default=None, help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--occ', action="store_true", help='whether to use occlusion')
    parser.add_argument('--load_kp', action="store_true", help='whether to use saved data')
    parser.add_argument('--load_skeleton', action="store_true", help='whether to use saved data')
    parser.add_argument('--random_skeleton', action="store_true", help='whether to use saved data')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # Load data
    support_img = cv2.imread(args.support)
    query_img = cv2.imread(args.query)
    if support_img is None or query_img is None:
        raise ValueError('Fail to read images')

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        Resize_Pad(cfg.model.encoder_config.img_size, cfg.model.encoder_config.img_size)])

    # frame = copy.deepcopy(support_img)
    padded_support_img = preprocess(support_img).cpu().numpy().transpose(1, 2, 0) * 255
    frame = copy.deepcopy(padded_support_img.astype(np.uint8).copy())
    kp_src = []
    skeleton = []
    count = 0
    prev_pt = None
    prev_pt_idx = None
    color_idx = 0

    def selectKP(event, x, y, flags, param):
        nonlocal kp_src, frame
        # if we are in points selection mode, the mouse was clicked,
        # list of  points with the (x, y) location of the click
        # and draw the circle

        if event == cv2.EVENT_LBUTTONDOWN:
            kp_src.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 1)
            cv2.imshow("Source", frame)

        if event == cv2.EVENT_RBUTTONDOWN:
            kp_src = []
            frame = copy.deepcopy(support_img)
            cv2.imshow("Source", frame)

    def draw_line(event, x, y, flags, param):
        nonlocal skeleton, kp_src, frame, count, prev_pt, prev_pt_idx, marked_frame, color_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            closest_point = min(kp_src, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
            closest_point_index = kp_src.index(closest_point)
            if color_idx < len(COLORS):
                c = COLORS[color_idx]
            else:
                c = random.choices(range(256), k=3)
            color = color_idx
            cv2.circle(frame, closest_point, 2, c, 1)
            if count == 0:
                prev_pt = closest_point
                prev_pt_idx = closest_point_index
                count = count + 1
                cv2.imshow("Source", frame)
            else:
                cv2.line(frame, prev_pt, closest_point, c, 2)
                cv2.imshow("Source", frame)
                count = 0
                skeleton.append((prev_pt_idx, closest_point_index))
                color_idx = color_idx + 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            frame = copy.deepcopy(marked_frame)
            cv2.imshow("Source", frame)
            count = 0
            color_idx = 0
            skeleton = []
            prev_pt = None

    if args.load_kp:
        with open("demo/kp.pkl", "rb") as fp:
            kp_src = pickle.load(fp)

    if args.load_skeleton:
        # skeleton = [
        #     [0, 2], [1, 2], [7, 4], [4, 3], [7, 6], [6, 5], [7, 8], [8, 9], [9, 10], [8, 12], [12, 11]
        # ]
        # skeleton = [
        #     [0, 1], [10, 11], [10, 3], [10, 5], [11, 3], [11, 5], [3, 5], [0, 2], [1, 2], [7, 4], [4, 3], [7, 6],
        #     [6, 5], [7, 8], [8, 9], [9, 10], [8, 12], [12, 11]
        # ]

        for i in range(1, 20):
            for j in range(i):
                skeleton.append((i, j))
        # skeleton = [(0, 0)]

        with open("demo/skeleton.npy", "rb") as fp:
            skeleton = pickle.load(fp)
        # skeleton += [
        #     [2, 12],
        # ]
        # skeleton = [(i, j) for i, j in skeleton]

    else:
        cv2.namedWindow("Source", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Source', 800, 600)
        cv2.setMouseCallback("Source", selectKP)
        cv2.imshow("Source", frame)

        # keep looping until points have been selected
        while len(kp_src) < 1:
            print('Press any key when finished marking the points!! ')
            cv2.waitKey(0)

        marked_frame = copy.deepcopy(frame)
        cv2.setMouseCallback("Source", draw_line)
        print('Press any key when finished creating skeleton!! ')
        while True:
            if cv2.waitKey(1) > 0:
                break
        # save keypoints and skeleton
        with open("demo/kp.pkl", "wb") as fp:
            pickle.dump(kp_src, fp)
        with open("demo/skeleton.npy", "wb") as fp:
            pickle.dump(skeleton, fp)


    skeleton += [
        [0, 2], [1, 2], [12, 2], [12, 7], [12, 8], [7, 3], [8, 4], [12, 11], [11, 9], [9, 5], [11, 10], [10, 6]
    ]
    skeleton = [(i, j) for i, j in skeleton]

    kp_src = torch.tensor(kp_src).float()
    # kp_src = transform_keypoints_to_pad_and_resize(kp_src, support_img.shape[:2])

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize_Pad(cfg.model.encoder_config.img_size, cfg.model.encoder_config.img_size)])

    num_runs = 1

    support_img = preprocess(support_img).flip(0)[None]
    query_img = preprocess(query_img).flip(0)[None]
    # Create heatmap from keypoints
    genHeatMap = TopDownGenerateTargetFewShot()
    data_cfg = cfg.data_cfg
    data_cfg['image_size'] = np.array([cfg.model.encoder_config.img_size, cfg.model.encoder_config.img_size])
    data_cfg['joint_weights'] = None
    data_cfg['use_different_joint_weights'] = False
    kp_src_3d = torch.concatenate((kp_src, torch.zeros(kp_src.shape[0], 1)), dim=-1)
    kp_src_3d_weight = torch.concatenate((torch.ones_like(kp_src), torch.zeros(kp_src.shape[0], 1)), dim=-1)
    target_s, target_weight_s = genHeatMap._msra_generate_target(data_cfg, kp_src_3d, kp_src_3d_weight, sigma=2)
    target_s = torch.tensor(target_s).float()[None]
    target_weight_s = torch.tensor(target_weight_s).float()[None]

    original_support_img = support_img.clone()
    if args.occ or args.random_skeleton:
        num_runs = 5

    for i in range(num_runs):
        if args.occ:
            patch_size = 96
            patch_location = (
                random.randint(0, cfg.model.encoder_config.img_size - patch_size),
                random.randint(0, cfg.model.encoder_config.img_size - patch_size))
            mask = torch.ones_like(support_img)
            mask[:, :, patch_location[0]:patch_location[0] + patch_size,
            patch_location[1]:patch_location[1] + patch_size] = 0
            support_img = original_support_img * mask
        if args.random_skeleton and i > 0:
            skeleton = shuffle_skeleton(skeleton)

        data = {
            'img_s': [support_img.cuda()],
            'img_q': query_img.cuda(),
            'target_s': [target_s.cuda()],
            'target_weight_s': [target_weight_s.cuda()],
            'target_q': None,
            'target_weight_q': None,
            'return_loss': False,
            'img_metas': [{'sample_skeleton': [skeleton],
                           'query_skeleton': skeleton,
                           'sample_joints_3d': [kp_src_3d.cuda()],
                           'query_joints_3d': kp_src_3d.cuda(),
                           'sample_center': [kp_src.mean(dim=0)],
                           'query_center': kp_src.mean(dim=0),
                           'sample_scale': [kp_src.max(dim=0)[0] - kp_src.min(dim=0)[0]],
                           'query_scale': kp_src.max(dim=0)[0] - kp_src.min(dim=0)[0],
                           'sample_rotation': [0],
                           'query_rotation': 0,
                           'sample_bbox_score': [1],
                           'query_bbox_score': 1,
                           'query_image_file': '',
                           'sample_image_file': [''],
                           }]
        }

        # Load model
        model = build_posenet(cfg.model).cuda()
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)
        model.eval()

        with torch.no_grad():
            outputs = model(**data)

        # visualize results
        vis_s_weight = target_weight_s[0]
        vis_q_weight = target_weight_s[0]
        vis_s_image = support_img[0].detach().cpu().numpy().transpose(1, 2, 0)
        vis_q_image = query_img[0].detach().cpu().numpy().transpose(1, 2, 0)
        support_kp = kp_src_3d
        _, original_skeleton = model.keypoint_head.skeleton_head.adj_mx_from_edges(num_pts=outputs['points'].shape[2],
                                                                                skeleton=[skeleton],
                                                                                mask=target_weight_s.squeeze(-1).bool(),
                                                                                device=target_weight_s.device)
        skeleton = outputs['skeleton']

        # old_plot_results(vis_s_image,
        #              vis_q_image,
        #              support_kp,
        #              vis_s_weight,
        #              None,
        #              vis_q_weight,
        #              skeleton,
        #              None,
        #              torch.tensor(outputs['points']).squeeze(),
        #              out_dir='demo')
        plot_results(vis_s_image,
                     vis_q_image,
                     support_kp,
                     vis_s_weight,
                     None,
                     vis_s_weight,
                     # None,
                     skeleton,
                     None,
                     torch.tensor(outputs['points']).squeeze(),
                     # target_keypoints,
                     out_dir='demo',
                     # in_color='green',
                     original_skeleton=original_skeleton[0].cpu().numpy(),
                     img_alpha=1.0,
                     radius=3,
                     )


if __name__ == '__main__':
    main()
