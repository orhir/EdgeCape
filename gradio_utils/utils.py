import copy
import json
import random
import collections
import gradio as gr
import numpy as np
import psutil
import torch
from PIL import ImageDraw, Image, ImageEnhance
from matplotlib import pyplot as plt
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmpose.core import wrap_fp16_model
from mmpose.models import build_posenet
from torchvision import transforms
import matplotlib.patheffects as mpe

from EdgeCape import TopDownGenerateTargetFewShot
from demo import Resize_Pad
from EdgeCape.models import *


def process_img(support_image, global_state):
    global_state['images']['image_orig'] = support_image
    if global_state["load_example"]:
        global_state["load_example"] = False
        return global_state['images']['image_kp'], global_state
    _, _ = reset_kp(global_state)
    return support_image, global_state


def adj_mx_from_edges(num_pts, skeleton, device='cuda', normalization_fix=True):
    adj_mx = torch.empty(0, device=device)
    batch_size = len(skeleton)
    for b in range(batch_size):
        edges = torch.tensor(skeleton[b]).long()
        adj = torch.zeros(num_pts, num_pts, device=device)
        adj[edges[:, 0], edges[:, 1]] = 1
        adj_mx = torch.concatenate((adj_mx, adj.unsqueeze(0)), dim=0)
    trans_adj_mx = torch.transpose(adj_mx, 1, 2)
    cond = (trans_adj_mx > adj_mx).float()
    adj = adj_mx + trans_adj_mx * cond - adj_mx * cond
    return adj

def plot_results(support_img, query_img, support_kp, support_w, query_kp, query_w,
                 skeleton=None, prediction=None, radius=6, in_color=None,
                 original_skeleton=None, img_alpha=0.6, target_keypoints=None):
    h, w, c = support_img.shape
    prediction = prediction[-1] * h
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(original_skeleton, list):
        original_skeleton = adj_mx_from_edges(num_pts=100, skeleton=[skeleton]).cpu().numpy()[0]
    query_img = (query_img - np.min(query_img)) / (np.max(query_img) - np.min(query_img))
    img = query_img
    w = query_w
    keypoint = prediction
    adj = skeleton
    color = None
    f, axes = plt.subplots()
    plt.imshow(img, alpha=img_alpha)
    for k in range(keypoint.shape[0]):
        if w[k] > 0:
            kp = keypoint[k, :2]
            c = (1, 0, 0, 0.75) if w[k] == 1 else (0, 0, 1, 0.6)
            patch = plt.Circle(kp,
                               radius,
                               color=c,
                               path_effects=[mpe.withStroke(linewidth=2, foreground='black')],
                               zorder=200)
            axes.add_patch(patch)
            axes.text(kp[0], kp[1], k, fontsize=(radius + 4), color='white', ha="center", va="center",
                      zorder=300,
                      path_effects=[
                          mpe.withStroke(linewidth=max(1, int((radius + 4) / 5)), foreground='black')])
            plt.draw()

    if adj is not None:
        max_skel_val = np.max(adj)
        draw_skeleton = adj / max_skel_val * 6
        for i in range(1, keypoint.shape[0]):
            for j in range(0, i):
                if w[i] > 0 and w[j] > 0 and original_skeleton[i][j] > 0:
                    if color is None:
                        num_colors = int((adj > 0.05).sum() / 2)
                        color = iter(plt.cm.rainbow(np.linspace(0, 1, num_colors + 1)))
                        c = next(color)
                    elif isinstance(color, str):
                        c = color
                    elif isinstance(color, collections.Iterable):
                        c = next(color)
                    else:
                        raise ValueError("Color must be a string or an iterable")
                if w[i] > 0 and w[j] > 0 and adj[i][j] > 0:
                    width = draw_skeleton[i][j]
                    stroke_width = width + (width / 3)
                    patch = plt.Line2D([keypoint[i, 0], keypoint[j, 0]],
                                       [keypoint[i, 1], keypoint[j, 1]],
                                       linewidth=width, color=c, alpha=0.6,
                                       path_effects=[mpe.withStroke(linewidth=stroke_width, foreground='black')],
                                       zorder=1)
                    axes.add_artist(patch)

        plt.axis('off')  # command for hiding the axis.
        return plt

def process(query_img, state,
            cfg_path='configs/test/1shot_split1.py',
            checkpoint_path='ckpt/1shot_split1.pth'):
    print(state)
    device = print_memory_usage()
    cfg = Config.fromfile(cfg_path)
    width, height, _ = np.array(state['images']['image_orig']).shape
    kp_src_np = np.array(state['points']).copy().astype(np.float32)
    kp_src_np[:, 0] = kp_src_np[:, 0] / width * 256
    kp_src_np[:, 1] = kp_src_np[:, 1] / height * 256
    kp_src_np = kp_src_np.copy()
    kp_src_tensor = torch.tensor(kp_src_np).float()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize_Pad(256, 256)
    ])

    if len(state['skeleton']) == 0:
        state['skeleton'] = [(0, 0)]

    support_img = preprocess(state['images']['image_orig']).flip(0)[None]
    np_query = np.array(query_img)[:, :, ::-1].copy()
    q_img = preprocess(np_query).flip(0)[None]
    # Create heatmap from keypoints
    genHeatMap = TopDownGenerateTargetFewShot()
    data_cfg = cfg.data_cfg
    data_cfg['image_size'] = np.array([256, 256])
    data_cfg['joint_weights'] = None
    data_cfg['use_different_joint_weights'] = False
    kp_src_3d = torch.cat(
        (kp_src_tensor, torch.zeros(kp_src_tensor.shape[0], 1)), dim=-1)
    kp_src_3d_weight = torch.cat(
        (torch.ones_like(kp_src_tensor),
         torch.zeros(kp_src_tensor.shape[0], 1)), dim=-1)
    target_s, target_weight_s = genHeatMap._msra_generate_target(data_cfg,
                                                                 kp_src_3d,
                                                                 kp_src_3d_weight,
                                                                 sigma=1)
    target_s = torch.tensor(target_s).float()[None]
    target_weight_s = torch.ones_like(
        torch.tensor(target_weight_s).float()[None])

    data = {
        'img_s': [support_img.to(device)],
        'img_q': q_img.to(device),
        'target_s': [target_s.to(device)],
        'target_weight_s': [target_weight_s.to(device)],
        'target_q': None,
        'target_weight_q': None,
        'return_loss': False,
        'img_metas': [{'sample_skeleton': [state['skeleton']],
                       'query_skeleton': state['skeleton'],
                       'sample_joints_3d': [kp_src_3d.to(device)],
                       'query_joints_3d': kp_src_3d.to(device),
                       'sample_center': [kp_src_tensor.mean(dim=0)],
                       'query_center': kp_src_tensor.mean(dim=0),
                       'sample_scale': [
                           kp_src_tensor.max(dim=0)[0] -
                           kp_src_tensor.min(dim=0)[0]
                       ],
                       'query_scale': kp_src_tensor.max(dim=0)[0] -
                                      kp_src_tensor.min(dim=0)[0],
                       'sample_rotation': [0],
                       'query_rotation': 0,
                       'sample_bbox_score': [1],
                       'query_bbox_score': 1,
                       'query_image_file': '',
                       'sample_image_file': [''],
                       }]
    }
    # Load model
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.eval().to(device)
    with torch.no_grad():
        outputs = model(**data)
    # visualize results
    vis_s_weight = target_weight_s[0]
    vis_s_image = support_img[0].detach().cpu().numpy().transpose(1, 2, 0)
    vis_q_image = q_img[0].detach().cpu().numpy().transpose(1, 2, 0)
    support_kp = kp_src_3d
    out = plot_results(vis_s_image,
                       vis_q_image,
                       support_kp,
                       vis_s_weight,
                       None,
                       vis_s_weight,
                       outputs['skeleton'][1],
                       torch.tensor(outputs['points']).squeeze(),
                       original_skeleton=state['skeleton'],
                       img_alpha=1.0,
                       )
    return out


def update_examples(support_img, query_image, global_state_str):
    example_state = json.loads(global_state_str)
    example_state["load_example"] = True
    example_state["curr_type_point"] = "start"
    example_state["prev_point"] = None
    example_state['images'] = {}
    example_state['images']['image_orig'] = support_img
    example_state['images']['image_kp'] = support_img
    example_state['images']['image_skeleton'] = support_img
    image_draw = example_state['images']['image_orig'].copy()
    for xy in example_state['points']:
        image_draw = update_image_draw(
            image_draw,
            xy,
            example_state
        )
    kp_image = image_draw.copy()
    example_state['images']['image_kp'] = kp_image
    pts_list = example_state['points']
    for limb in example_state['skeleton']:
        prev_point = pts_list[limb[0]]
        curr_point = pts_list[limb[1]]
        points = [prev_point, curr_point]
        image_draw = draw_limbs_on_image(image_draw,
                                         points
                                         )
    skel_image = image_draw.copy()
    example_state['images']['image_skel'] = skel_image
    return (support_img,
            kp_image,
            skel_image,
            query_image,
            example_state)


def get_select_coords(global_state,
                      evt: gr.SelectData
                      ):
    """This function only support click for point selection
    """
    xy = evt.index
    global_state["points"].append(xy)
    image_raw = global_state['images']['image_kp']
    image_draw = update_image_draw(
        image_raw,
        xy,
        global_state
    )
    global_state['images']['image_kp'] = image_draw
    return global_state, image_draw

def get_closest_point_idx(pts_list, xy):
    x, y = xy
    closest_point = min(pts_list, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
    closest_point_index = pts_list.index(closest_point)
    return closest_point_index


def reset_skeleton(global_state):
    image = global_state["images"]["image_kp"]
    global_state["images"]["image_skel"] = image
    global_state["skeleton"] = []
    global_state["curr_type_point"] = "start"
    global_state["prev_point"] = None
    return image


def reset_kp(global_state):
    image = global_state["images"]["image_orig"]
    global_state["images"]["image_kp"] = image
    global_state["images"]["image_skel"] = image
    global_state["skeleton"] = []
    global_state["points"] = []
    global_state["curr_type_point"] = "start"
    global_state["prev_point"] = None
    return image, image


def select_skeleton(global_state,
                    evt: gr.SelectData,
                    ):
    xy = evt.index
    pts_list = global_state["points"]
    closest_point_idx = get_closest_point_idx(pts_list, xy)
    image_raw = global_state['images']['image_skel']
    if global_state["curr_type_point"] == "end":
        prev_point_idx = global_state["prev_point_idx"]
        prev_point = pts_list[prev_point_idx]
        points = [prev_point, xy]
        image_draw = draw_limbs_on_image(image_raw,
                                         points
                                         )
        global_state['images']['image_skel'] = image_draw
        global_state['skeleton'].append([prev_point_idx, closest_point_idx])
        global_state["curr_type_point"] = "start"
        global_state["prev_point_idx"] = None
    else:
        global_state["prev_point_idx"] = closest_point_idx
        global_state["curr_type_point"] = "end"
    return global_state, global_state['images']['image_skel']


def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points


def update_image_draw(image, points, global_state):
    if len(global_state["points"]) < 2:
        alpha = 0.5
    else:
        alpha = 1.0
    image_draw = draw_points_on_image(image, points, alpha=alpha)
    return image_draw


def print_memory_usage():
    # Print system memory usage
    print(f"System memory usage: {psutil.virtual_memory().percent}%")

    # Print GPU memory usage
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1e9} GB")
        print(
            f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9} GB")
        device_properties = torch.cuda.get_device_properties(device)
        available_memory = device_properties.total_memory - \
                           torch.cuda.max_memory_allocated()
        print(f"Available GPU memory: {available_memory / 1e9} GB")
    else:
        device = "cpu"
        print("No GPU available")
    return device

def draw_limbs_on_image(image,
                        points,):
    color = tuple(random.choices(range(256), k=3))
    overlay_rgba = Image.new("RGBA", image.size, 0)
    overlay_draw = ImageDraw.Draw(overlay_rgba)
    p_start, p_target = points
    if p_start is not None and p_target is not None:
        p_draw = int(p_start[0]), int(p_start[1])
        t_draw = int(p_target[0]), int(p_target[1])
        overlay_draw.line(
            (p_draw[0], p_draw[1], t_draw[0], t_draw[1]),
            fill=color,
            width=10,
        )

    return Image.alpha_composite(image.convert("RGBA"),
                                 overlay_rgba).convert("RGB")


def draw_points_on_image(image,
                         points,
                         radius_scale=0.01,
                         alpha=1.):
    if alpha < 1:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
    overlay_rgba = Image.new("RGBA", image.size, 0)
    overlay_draw = ImageDraw.Draw(overlay_rgba)
    p_color = (255, 0, 0)
    rad_draw = int(image.size[0] * radius_scale)
    if points is not None:
        p_draw = int(points[0]), int(points[1])
        overlay_draw.ellipse(
            (
                p_draw[0] - rad_draw,
                p_draw[1] - rad_draw,
                p_draw[0] + rad_draw,
                p_draw[1] + rad_draw,
            ),
            fill=p_color,
            )

    return Image.alpha_composite(image.convert("RGBA"), overlay_rgba).convert("RGB")
