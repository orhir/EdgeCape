import collections
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
import torch.nn.functional as F
import uuid

from matplotlib.colors import BoundaryNorm
import matplotlib.patheffects as mpe
from itertools import cycle

colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def plot_heatmap(support_img, query_img, heatmaps, support_kp, support_w, query_kp, query_w, skeleton,
                 initial_proposals, prediction, radius=6, n_heatmaps=5):
    h, w, c = support_img.shape
    fig, axes = plt.subplots(n_heatmaps + 1, 4, gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.set_size_inches(40, 10 * (n_heatmaps - 1), forward=True)
    [axi.set_axis_off() for axi in axes.ravel()]
    plt.subplots_adjust(wspace=0, hspace=0)
    # Plot Skeleton
    support_img = (support_img - np.min(support_img)) / (np.max(support_img) - np.min(support_img))
    query_img = (query_img - np.min(query_img)) / (np.max(query_img) - np.min(query_img))
    axes[0, 0].imshow(support_img)
    axes[0, 1].imshow(query_img)
    axes[0, 2].imshow(support_img)
    axes[0, 3].imshow(query_img)
    for k in range(support_kp.shape[0]):
        if support_w[k] > 0:
            kp = support_kp[k, :2]
            c = (1, 0, 0, 0.75) if support_w[k] == 1 else (0, 0, 1, 0.6)
            patch = plt.Circle(kp, radius, color=c)
            axes[0, 0].add_patch(patch)
            axes[0, 0].text(kp[0], kp[1], k)
            kp = query_kp[k, :2]
            c = (1, 0, 0, 0.75) if query_kp[k, 2] == 1 else (0, 0, 1, 0.6)
            patch = plt.Circle(kp, radius, color=c)
            axes[0, 1].add_patch(patch)
            axes[0, 1].text(kp[0], kp[1], k)
            plt.draw()
    for l, limb in enumerate(skeleton):
        if l > len(colors) - 1:
            c = [x / 255 for x in random.sample(range(0, 255), 3)]
        else:
            c = [x / 255 for x in colors[l]]
        if support_w[limb[0]] > 0 and support_w[limb[1]] > 0 and query_w[limb[0]] > 0 and query_w[limb[1]] > 0:
            patch = plt.Line2D([support_kp[limb[0], 0], support_kp[limb[1], 0]],
                               [support_kp[limb[0], 1], support_kp[limb[1], 1]],
                               linewidth=2, color=c, alpha=0.5)
            axes[0, 2].add_artist(patch)
            patch = plt.Line2D([query_kp[limb[0], 0], query_kp[limb[1], 0]],
                               [query_kp[limb[0], 1], query_kp[limb[1], 1]],
                               linewidth=2, color=c, alpha=0.5)
            axes[0, 3].add_artist(patch)
    # Plot heatmap
    prediction = prediction[-1] * h
    initial_proposals = initial_proposals[0] * h
    # similarity_map = F.interpolate(heatmaps[:, None], size=(h, w), mode='bilinear').squeeze()
    similarity_map = heatmaps
    # similarity_map_shape = similarity_map.shape
    # similarity_map = similarity_map.reshape(*similarity_map_shape[:2], -1)
    # similarity_map = (similarity_map - torch.min(
    #     similarity_map, dim=2)[0].unsqueeze(2)) / (
    #                          torch.max(similarity_map, dim=2)[0].unsqueeze(2) -
    #                          torch.min(similarity_map, dim=2)[0].unsqueeze(2) + 1e-10)
    j = 0
    for i in range(n_heatmaps):
        if support_w[j] > 0 and query_w[j] > 0:
            if i > len(colors) - 1:
                c = [x / 255 for x in random.sample(range(0, 255), 3)]
            else:
                c = [x / 255 for x in colors[i]]
            kp = support_kp[j, :2]
            patch = plt.Circle(kp, radius, color=c, alpha=0.6)
            axes[i + 1, 0].add_patch(patch)
            axes[i + 1, 0].text(kp[0], kp[1], j)
            axes[i + 1, 0].imshow(support_img)
            axes[i + 1, 1].imshow(similarity_map[j].cpu().numpy(), alpha=0.6, cmap='jet')
            axes[i + 1, 2].imshow(query_img)
            patch = plt.Circle(initial_proposals[j], 0.2 * h, color=c, alpha=0.6)
            axes[i + 1, 2].add_patch(patch)
            patch = plt.Circle(query_kp[j], radius, color=(1, 0, 0), alpha=0.8)
            axes[i + 1, 2].add_patch(patch)
            axes[i + 1, 2].text(initial_proposals[j][0], initial_proposals[j][1], j)
            axes[i + 1, 3].imshow(query_img)
            patch = plt.Circle(prediction[j], 0.2 * h, color=c, alpha=0.6)
            axes[i + 1, 3].add_patch(patch)
            patch = plt.Circle(query_kp[j], radius, color=(1, 0, 0), alpha=0.8)
            axes[i + 1, 3].add_patch(patch)
            axes[i + 1, 3].text(initial_proposals[j][0], initial_proposals[j][1], j)
        j += 1
        if j > 99:
            break
    img_names = [img.split(".")[0] for img in os.listdir('./heatmaps') if img.endswith('.png')]
    if len(img_names) > 0:
        name_idx = max([int(img_name) for img_name in img_names]) + 1
    else:
        name_idx = 0
    plt.savefig(f'./heatmaps/{str(name_idx)}.png')
    plt.clf()


def plot_attn(support_img, query_img, similarity_map, support_kp, support_w, query_kp, query_w, skeleton,
              attn_map, adjs, prediction, radius=14, n_heatmaps=1):
    h, w, c = support_img.shape
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    fig, axes = plt.subplots(4, 4, gridspec_kw={'wspace': 0.2, 'hspace': 0.2})
    fig.set_size_inches(50, 50, forward=True)
    axes[0, 0].set_axis_off()
    axes[0, 1].set_axis_off()
    axes[0, 2].set_axis_off()
    axes[0, 3].set_axis_off()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # Plot Skeleton
    support_img = (support_img - np.min(support_img)) / (np.max(support_img) - np.min(support_img))
    query_img = (query_img - np.min(query_img)) / (np.max(query_img) - np.min(query_img))
    axes[0, 0].imshow(support_img)
    axes[0, 1].imshow(query_img)
    axes[0, 2].imshow(support_img)
    axes[0, 3].imshow(query_img)
    for k in range(support_kp.shape[0]):
        if support_w[k] > 0:
            kp = support_kp[k, :2]
            c = (1, 0, 0, 0.75) if support_w[k] == 1 else (0, 0, 1, 0.6)
            patch = plt.Circle(kp, radius, color=c)
            axes[0, 0].add_patch(patch)
            axes[0, 0].text(kp[0], kp[1], k)
            kp = query_kp[k, :2]
            c = (1, 0, 0, 0.75) if query_kp[k, 2] == 1 else (0, 0, 1, 0.6)
            patch = plt.Circle(kp, radius, color=c)
            axes[0, 1].add_patch(patch)
            axes[0, 1].text(kp[0], kp[1], k)
            plt.draw()
    for l, limb in enumerate(skeleton):
        if l > len(colors) - 1:
            c = [x / 255 for x in random.sample(range(0, 255), 3)]
        else:
            c = [x / 255 for x in colors[l]]
        if support_w[limb[0]] > 0 and support_w[limb[1]]:
            patch = plt.Line2D([support_kp[limb[0], 0], support_kp[limb[1], 0]],
                               [support_kp[limb[0], 1], support_kp[limb[1], 1]],
                               linewidth=8, color=c, alpha=0.5)
            axes[0, 2].add_artist(patch)
        if query_w[limb[0]] > 0 and query_w[limb[1]]:
            patch = plt.Line2D([query_kp[limb[0], 0], query_kp[limb[1], 0]],
                               [query_kp[limb[0], 1], query_kp[limb[1], 1]],
                               linewidth=8, color=c, alpha=0.5)
            axes[0, 3].add_artist(patch)
    # Plot heatmap
    axes[1, 0].set_title("GT")
    axes[1, 1].set_title("L1")
    axes[1, 2].set_title("L2")
    axes[1, 3].set_title("L3")
    min_kp_pos = np.argmax(np.cumsum(query_w)) + 1
    mask = torch.from_numpy(query_w).bool()[None]
    gt_A = adj_mx_from_edges(num_pts=100, skeleton=[skeleton], device=mask.device).cpu().numpy()
    gt_A = gt_A[:min_kp_pos, :min_kp_pos]
    axes[1, 0].imshow(gt_A, alpha=0.6, cmap='Reds')
    for i in range(min_kp_pos):
        for j in range(min_kp_pos):
            text = axes[1, 0].text(j, i, np.round(gt_A[i, j], 2), ha="center", va="center")
    np.fill_diagonal(gt_A, 0)
    axes[2, 0].imshow(gt_A, alpha=0.6, cmap='Reds')
    for i in range(min_kp_pos):
        for j in range(min_kp_pos):
            text = axes[2, 0].text(j, i, np.round(gt_A[i, j], 2), ha="center", va="center")
    for col, attn in enumerate(attn_map):
        heatmap = attn[:, :min_kp_pos, :min_kp_pos].squeeze().cpu().numpy()
        axes[1, col+1].imshow(heatmap, alpha=0.6, cmap='Reds')
        for i in range(min_kp_pos):
            for j in range(min_kp_pos):
                text = axes[1, col+1].text(j, i, np.round(heatmap[i, j], 2), ha="center", va="center")
        # np.fill_diagonal(heatmap, 0)
        # heatmap = heatmap / heatmap.sum(1, keepdims=True)
        axes[2, col+1].imshow(heatmap, alpha=0.6, cmap='Reds')
        for i in range(min_kp_pos):
            for j in range(min_kp_pos):
                text = axes[2, col+1].text(j, i, np.round(heatmap[i, j], 2), ha="center", va="center")

        # Plot self-attention on image
        self_attention_skeleton = []
        for i in range(min_kp_pos):
            topk = np.argsort(heatmap[i])[::-1]
            for m in range(5):
                self_attention_skeleton.append([i, topk[m], heatmap[i, topk[m]]])
        axes[3, col+1].imshow(query_img)
        for k in range(support_kp.shape[0]):
            if support_w[k] > 0:
                kp = query_kp[k, :2]
                c = (1, 0, 0, 0.75) if query_kp[k, 2] == 1 else (0, 0, 1, 0.6)
                patch = plt.Circle(kp, radius//2, color=c)
                axes[3, col+1].add_patch(patch)
                axes[3, col+1].text(kp[0], kp[1], k, fontsize=12)
                plt.draw()
        for l, limb in enumerate(self_attention_skeleton):
            if query_w[limb[0]] > 0 and query_w[limb[1]]:
                patch = plt.Line2D(
                    [query_kp[limb[0], 0], query_kp[limb[1], 0]],
                    [query_kp[limb[0], 1], query_kp[limb[1], 1]],
                    linewidth=30*limb[2], color='red', alpha=limb[2])
                axes[3, col+1].add_artist(patch)
        # cur_adj = torch.nn.functional.sigmoid(adjs[col])[:, :min_kp_pos, :min_kp_pos].squeeze().cpu().numpy()
        # axes[3, col + 1].imshow(cur_adj, alpha=0.6, cmap='Reds')
        # for i in range(min_kp_pos):
        #     for j in range(min_kp_pos):
        #         text = axes[3, col + 1].text(j, i, np.round(cur_adj[i, j], 2), ha="center", va="center")
    img_names = [img.split(".")[0] for img in os.listdir('./heatmaps') if str_is_int(img.split(".")[0])]
    if len(img_names) > 0:
        name_idx = max([int(img_name) for img_name in img_names]) + 1
    else:
        name_idx = 0
    # crete dir
    # if not os.path.isdir(f'./heatmaps/{str(name_idx)}'):
    #     os.mkdir(f'./heatmaps/{str(name_idx)}')
    plt.savefig(f'./heatmaps/{str(name_idx)}.png')
    extent = axes[3,3].get_window_extent().transformed(
        fig.dpi_scale_trans.inverted())
    fig.savefig(f'./heatmaps/layer_{str(name_idx)}.png', bbox_inches=extent)
    # for k, row in enumerate(axes):
    #     for i, ax in enumerate(row):
    #         extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #         plt.savefig(f'./heatmaps/{str(name_idx)}/{str(k)}_{str(i)}.png', bbox_inches=extent)

    plt.clf()


def plot_results(support_img, query_img, support_kp, support_w, query_kp, query_w,
                 skeleton=None, initial_proposals=None, prediction=None,
                 radius=6, out_dir='./heatmaps', file_name=None, in_color=None,
                 original_skeleton=None, img_alpha=0.6, target_keypoints=None):
    img_names = [img.split("_")[0] for img in os.listdir(out_dir) if str_is_int(img.split("_")[0])]
    if file_name is None:
        if len(img_names) > 0:
            name_idx = str(max([int(img_name) for img_name in img_names]) + 1)
        else:
            name_idx = '0'
    else:
        name_idx = file_name
    # crete dir
    # if not os.path.isdir(f'./heatmaps/{str(name_idx)}'):
    #     os.mkdir(f'./heatmaps/{str(name_idx)}')

    h, w, c = support_img.shape
    prediction = prediction[-1] * h
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(skeleton, list):
        skeleton = adj_mx_from_edges(num_pts=100, skeleton=[skeleton]).cpu().numpy()[0]
        original_skeleton = skeleton
    support_img = (support_img - np.min(support_img)) / (np.max(support_img) - np.min(support_img))
    query_img = (query_img - np.min(query_img)) / (np.max(query_img) - np.min(query_img))
    error_mask = None
    for id, (img, w, keypoint, adj) in enumerate(zip([support_img, support_img, query_img],
                                                [support_w, support_w, query_w],
                                                # [support_kp, query_kp])):
                                                [support_kp, support_kp, prediction],
                                                [original_skeleton, skeleton, skeleton])):
        color = in_color
        f, axes = plt.subplots()
        plt.imshow(img, alpha=img_alpha)

        # On qeury image plot
        if id == 2 and target_keypoints is not None:
            error = np.linalg.norm(keypoint - target_keypoints, axis=-1)
            error_mask = error > (256 * 0.05)

        for k in range(keypoint.shape[0]):
            if w[k] > 0:
                kp = keypoint[k, :2]
                c = (1, 0, 0, 0.75) if w[k] == 1 else (0, 0, 1, 0.6)
                if error_mask is not None and error_mask[k]:
                    c = (1, 1, 0, 0.75)
                    patch = plt.Circle(kp,
                                       radius,
                                       color=c,
                                       path_effects=[mpe.withStroke(linewidth=8, foreground='black'),
                                                     mpe.withStroke(linewidth=4, foreground='white'),
                                                     mpe.withStroke(linewidth=2, foreground='black'),
                                                     ],
                                       zorder=260)
                    axes.add_patch(patch)
                    axes.text(kp[0], kp[1], k, fontsize=10, color='black', ha="center", va="center", zorder=320,)
                else:
                    patch = plt.Circle(kp,
                                       radius,
                                       color=c,
                                       path_effects=[mpe.withStroke(linewidth=2, foreground='black')],
                                       zorder=200)
                    axes.add_patch(patch)
                    axes.text(kp[0], kp[1], k, fontsize=(radius+4), color='white', ha="center", va="center", zorder=300,
                              path_effects=[mpe.withStroke(linewidth=max(1, int((radius+4)/5)), foreground='black')])
                # axes.text(kp[0], kp[1], k)
                plt.draw()
        # Create keypoint pairs index list
        # color_hack = {
        #     (0, 1): '3000ff',
        #     (1, 2): 'ff008a',
        #     (2, 3): 'ff00de',
        #     (3, 4): 'd200ff',
        #     (4, 5): '8400ff',
        #     (5, 0): '003cff',
        # }
        # reverse_key_color_hack = {(k[1], k[0]): v for k, v in color_hack.items()}
        # color_hack = {**color_hack, **reverse_key_color_hack}
        if adj is not None:
            # Make max value 6
            draw_skeleton = adj ** 1
            max_skel_val = np.max(draw_skeleton)
            draw_skeleton = draw_skeleton / max_skel_val * 6
            for i in range(1, keypoint.shape[0]):
                for j in range(0, i):
                    # if c_index > len(colors) - 1:
                    #     c = [x / 255 for x in random.sample(range(0, 255), 3)]
                    # else:
                    #     c = [x / 255 for x in colors[c_index]]
                    # if (i, j) in color_hack:
                    #     c = color_hack[(i, j)]
                    #     c = [int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)]
                    #     c_index -= 1
                    if w[i] > 0 and w[j] > 0 and original_skeleton[i][j] > 0:
                        if color is None:
                            num_colors = int((skeleton > 0.05).sum() / 2)
                            color = iter(plt.cm.rainbow(np.linspace(0, 1, num_colors+1)))
                            c = next(color)
                        elif isinstance(color, str):
                            c = color
                        elif isinstance(color, collections.Iterable):
                            c = next(color)
                        else:
                            raise ValueError("Color must be a string or an iterable")
                    if w[i] > 0 and w[j] > 0 and skeleton[i][j] > 0:
                        width = draw_skeleton[i][j]
                        stroke_width = width + (width / 3)
                        patch = plt.Line2D([keypoint[i, 0], keypoint[j, 0]],
                                           [keypoint[i, 1], keypoint[j, 1]],
                                           linewidth=width, color=c, alpha=0.6,
                                           path_effects=[mpe.withStroke(linewidth=stroke_width, foreground='black')], zorder=1)
                        axes.add_artist(patch)

        plt.axis('off')  # command for hiding the axis.
        plt.savefig(f'./{out_dir}/{str(name_idx)}_{str(id)}.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        # plt.close('all')


def old_plot_results(support_img, query_img, support_kp, support_w, query_kp, query_w, skeleton,
                 initial_proposals, prediction, radius=6, out_dir='./heatmaps',
                 file_name=None):
    img_names = [img.split("_")[0] for img in os.listdir(out_dir) if str_is_int(img.split("_")[0])]
    if file_name is None:
        if len(img_names) > 0:
            name_idx = str(max([int(img_name) for img_name in img_names]) + 1)
        else:
            name_idx = '0'
    else:
        name_idx = file_name
    # crete dir
    # if not os.path.isdir(f'./heatmaps/{str(name_idx)}'):
    #     os.mkdir(f'./heatmaps/{str(name_idx)}')

    h, w, c = support_img.shape
    prediction = prediction[-1].cpu().numpy() * h
    support_img = (support_img - np.min(support_img)) / (np.max(support_img) - np.min(support_img))
    query_img = (query_img - np.min(query_img)) / (np.max(query_img) - np.min(query_img))

    for id, (img, w, keypoint) in enumerate(zip([support_img, query_img],
                                                [support_w, query_w],
                                                # [support_kp, query_kp])):
                                                [support_kp, prediction])):
        f, axes = plt.subplots()
        plt.imshow(img)
        for k in range(keypoint.shape[0]):
            if w[k] > 0:
                kp = keypoint[k, :2]
                c = (1, 0, 0, 0.75) if w[k] == 1 else (0, 0, 1, 0.6)
                patch = plt.Circle(kp, radius, color=c)
                axes.add_patch(patch)
                axes.text(kp[0], kp[1], k, fontsize=20)
                # axes.text(kp[0], kp[1], k)
                plt.draw()
        # Create keypoint pairs index list
        # color_hack = {
        #     (0, 1): '3000ff',
        #     (1, 2): 'ff008a',
        #     (2, 3): 'ff00de',
        #     (3, 4): 'd200ff',
        #     (4, 5): '8400ff',
        #     (5, 0): '003cff',
        # }
        # reverse_key_color_hack = {(k[1], k[0]): v for k, v in color_hack.items()}
        # color_hack = {**color_hack, **reverse_key_color_hack}
        # c_index = 0
        # for i in range(1, keypoint.shape[0]):
        #     for j in range(0, i):
        #         if c_index > len(colors) - 1:
        #             c = [x / 255 for x in random.sample(range(0, 255), 3)]
        #         else:
        #             c = [x / 255 for x in colors[c_index]]
        #         if (i, j) in color_hack:
        #             c = color_hack[(i, j)]
        #             c = [int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        #             c_index -= 1
        #         if w[i] > 0 and w[j] > 0 and skeleton[i][j] > 0:
        #             patch = plt.Line2D([keypoint[i, 0], keypoint[j, 0]],
        #                                [keypoint[i, 1], keypoint[j, 1]],
        #                                # linewidth=skeleton[i][j]*20, color=c, alpha=0.6)
        #                                linewidth=5, color=c, alpha=0.6)
        #             axes.add_artist(patch)
        #             c_index += 1

        for l, limb in enumerate(skeleton):
            kp = keypoint[:, :2]
            if l > len(colors) - 1:
                c = [x / 255 for x in random.sample(range(0, 255), 3)]
            else:
                c = [x / 255 for x in colors[l]]
            if w[limb[0]] > 0 and w[limb[1]] > 0:
                patch = plt.Line2D([kp[limb[0], 0], kp[limb[1], 0]],
                                   [kp[limb[0], 1], kp[limb[1], 1]],
                                   linewidth=6, color=c, alpha=0.6)
                axes.add_artist(patch)
        plt.axis('off')  # command for hiding the axis.
        plt.savefig(f'./{out_dir}/{str(name_idx)}_{str(id)}.png', bbox_inches='tight', pad_inches=0)
        plt.clf()

def str_is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def adj_mx_from_edges(num_pts, skeleton, device='cuda', normalization_fix=True):
    adj_mx = torch.empty(0, device=device)
    batch_size = len(skeleton)
    for b in range(batch_size):
        edges = torch.tensor(skeleton[b])
        adj = torch.zeros(num_pts, num_pts, device=device)
        adj[edges[:, 0], edges[:, 1]] = 1
        adj_mx = torch.concatenate((adj_mx, adj.unsqueeze(0)), dim=0)
    trans_adj_mx = torch.transpose(adj_mx, 1, 2)
    cond = (trans_adj_mx > adj_mx).float()
    adj = adj_mx + trans_adj_mx * cond - adj_mx * cond
    # if normalization_fix:
    #     adj = adj * ~mask[..., None] * ~mask[:, None]
    #     adj = torch.nan_to_num(adj / adj.sum(dim=-1, keepdim=True))
    # else:
    #     adj = torch.nan_to_num(adj / adj.sum(dim=2, keepdim=True)) * ~mask[..., None] * ~mask[:, None]
    # adj = torch.stack((torch.diag_embed(~mask), adj), dim=1)
    return adj

def vis_skeleton(support_img, support_kp, support_w, a_pred, a_gt, file_name=None, radius=3, line_width=6, alpha=0.8):
    h, w, c = support_img.shape
    # Normalize the support image
    support_img = (support_img - np.min(support_img)) / (np.max(support_img) - np.min(support_img))
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), gridspec_kw={'height_ratios': [1, 1]})
    axes[0, 0].imshow(support_img, alpha=0.6)
    axes[0, 1].imshow(support_img, alpha=0.6)
    axes[1, 0].imshow(support_img, alpha=0.6)
    axes[1, 1].axis('off')  # Hide the unused subplot

    a_pred = (a_pred + a_pred.T) / 2
    scaled_a_pred = a_pred ** 1 * line_width
    max_val = np.max(scaled_a_pred)
    for i in range(a_pred.shape[0]):
        for j in range(i + 1, a_pred.shape[1]):
            if support_w[i] > 0 and support_w[j] > 0 and a_pred[i, j] > 0:
                kp1 = support_kp[i, :2]
                kp2 = support_kp[j, :2]
                width = scaled_a_pred[i, j]
                stroke_width = width + (width / 3)
                outline = mpe.withStroke(linewidth=stroke_width, foreground='black')
                patch = plt.Line2D([kp1[0], kp2[0]], [kp1[1], kp2[1]],
                                   path_effects=[outline],
                                   linewidth=width,
                                   color='blue',
                                   alpha=alpha)
                axes[0, 0].add_artist(patch)

    # Plot keypoints and skeleton for predicted adjacency matrix
    for k in range(support_kp.shape[0]):
        if support_w[k] > 0:
            kp = support_kp[k, :2]
            outline = mpe.withStroke(linewidth=2, foreground='black')
            patch = plt.Circle(kp, radius, color=(1, 0, 0, 1), path_effects=[outline], zorder=200)
            axes[0, 0].add_patch(patch)

    a_gt = (a_gt + a_gt.T) / 2
    for i in range(a_gt.shape[0]):
        for j in range(i + 1, a_gt.shape[1]):
            if support_w[i] > 0 and support_w[j] > 0 and a_gt[i, j] > 0:
                kp1 = support_kp[i, :2]
                kp2 = support_kp[j, :2]
                width = a_gt[i, j] * max_val
                outline = mpe.withStroke(linewidth=width+2, foreground='black')
                patch = plt.Line2D([kp1[0], kp2[0]], [kp1[1], kp2[1]],
                                   path_effects=[outline],
                                   linewidth=width,
                                   color='green',
                                   alpha=alpha)
                axes[0, 1].add_artist(patch)

    # Plot keypoints and skeleton for predicted adjacency matrix
    for k in range(support_kp.shape[0]):
        if support_w[k] > 0:
            kp = support_kp[k, :2]
            outline = mpe.withStroke(linewidth=3, foreground='black')
            patch = plt.Circle(kp, radius, color=(1, 0, 0, 1), path_effects=[outline], zorder=200)
            axes[0, 1].add_patch(patch)
            # axes[0, 0].text(kp[0], kp[1],
            #                 k,
            #                 path_effects=[mpe.Stroke(linewidth=2, foreground='black'), mpe.Normal()],
            #                 fontsize=12,
            #                 color='white',
            #                 ha="center",
            #                 va="center",
            #                 zorder=300)

    # Calculate the difference and plot the skeleton with color based on the difference
    diff = (a_pred - a_gt) / (a_gt + 1e-10)
    for k in range(support_kp.shape[0]):
        if support_w[k] > 0:
            kp = support_kp[k, :2]
            patch = plt.Circle(kp, radius, color=(1, 0, 0, 0.75))
            axes[1, 0].add_patch(patch)
            axes[1, 0].text(kp[0], kp[1], k, fontsize=8)

    cmap = shiftedColorMap(plt.cm.Spectral, midpoint=0.34)
    norm = plt.Normalize(vmin=-1., vmax=2.)

    for i in range(diff.shape[0]):
        for j in range(i + 1, diff.shape[1]):
            if support_w[i] > 0 and support_w[j] > 0 and diff[i, j] != 0:
                kp1 = support_kp[i, :2]
                kp2 = support_kp[j, :2]
                color = cmap(norm(diff[i, j]))
                patch = plt.Line2D([kp1[0], kp2[0]], [kp1[1], kp2[1]],
                                   linewidth=line_width/2,
                                   color=color,
                                   alpha=alpha)
                axes[1, 0].add_artist(patch)

    # axes[0, 0].set_title('Predicted Adjacency Matrix')
    # axes[0, 1].set_title('Ground-Truth Adjacency Matrix')
    # axes[1, 0].set_title(r'$\frac{(a_{pred} - a_{gt})}{a_{gt}}$')

    for ax in axes[0, :]:
        ax.axis('off')
    for ax in axes[1, :]:
        ax.axis('off')

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes[1, 0], orientation='vertical')
    cbar.set_label('Difference')

    if file_name:
        path = f'./heatmaps/{file_name}'
        plt.savefig(f'{path}_pred.png', bbox_inches='tight', pad_inches=0)
        extent = axes[0, 0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{path}_prediction.png', bbox_inches=extent)
        extent = axes[0, 1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{path}_gt.png', bbox_inches=extent)
        extent = axes[1, 0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{path}_diff.png', bbox_inches=extent.expanded(1.6, 1.3))
    plt.cla()


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    # plt.register_cmap(cmap=newcmap)

    return newcmap