import math
import cv2
import mmcv
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from mmcv.image import imwrite
from mmcv.visualization.image import imshow
from mmpose.models import builder
from mmpose.models.builder import POSENETS
from mmpose.models.detectors.base import BasePose
from EdgeCape.models.backbones.adapter import DPT
from EdgeCape.models.backbones.dino import DINO


@POSENETS.register_module()
class EdgeCape(BasePose):
    """
    EdgeCape: Edge-aware Context-Aware Pose Estimation.
    Args:
        keypoint_head (dict): Config for keypoint head.
        encoder_config (dict): Config for encoder.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        freeze_backbone (bool): If True, freeze backbone. Default: False.
    """

    def __init__(self,
                 keypoint_head,
                 encoder_config,
                 train_cfg=None,
                 test_cfg=None,
                 freeze_backbone=False):
        super().__init__()
        feature_output_setting = encoder_config.get('output', 'dense-cls')
        model_name = encoder_config.get('model_name', 'vits14')
        self.encoder_sample = self.encoder_query = DINO(output=feature_output_setting, model_name=model_name)
        self.probe = DPT(input_dims=self.encoder_query.feat_dim, output_dim=768)
        self.backbone = 'dino_extractor'
        self.freeze_backbone = freeze_backbone
        if keypoint_head.get('freeze', None) is not None:
            self.freeze_backbone = True

        self.keypoint_head_module = builder.build_head(keypoint_head)
        self.keypoint_head_module.init_weights()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.target_type = test_cfg.get('target_type',
                                        'GaussianHeatMap')  # GaussianHeatMap

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head_module')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.encoder_sample.init_weights(pretrained)
        self.encoder_query.init_weights(pretrained)
        self.keypoint_head_module.init_weights()

    def forward(self,
                img_s,
                img_q,
                target_s=None,
                target_weight_s=None,
                target_q=None,
                target_weight_q=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img_s, target_s, target_weight_s, img_q,
                                      target_q, target_weight_q, img_metas,
                                      **kwargs)
        else:
            return self.forward_test(img_s, target_s, target_weight_s, img_q,
                                     target_q, target_weight_q, img_metas,
                                     **kwargs)

    def forward_train(self,
                      img_s,
                      target_s,
                      target_weight_s,
                      img_q,
                      target_q,
                      target_weight_q,
                      img_metas,
                      **kwargs):
        """Defines the computation performed at every call when training."""
        bs, _, h, w = img_q.shape
        random_mask = kwargs.get('rand_mask', None)
        output, initial_proposals, similarity_map, mask_s, reconstructed_keypoints = self.predict(img_s,
                                                                                                  target_s,
                                                                                                  target_weight_s,
                                                                                                  img_q,
                                                                                                  img_metas,
                                                                                                  random_mask)

        # parse the img meta to get the target keypoints
        device = output.device
        target_keypoints = self.parse_keypoints_from_img_meta(img_metas,
                                                              device,
                                                              keyword='query')

        target_sizes = torch.tensor(
            [img_q.shape[-2], img_q.shape[-1]]).unsqueeze(0).repeat(
            img_q.shape[0], 1, 1)

        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head_module.get_loss(output,
                                                                 initial_proposals,
                                                                 similarity_map,
                                                                 target_keypoints,
                                                                 target_q,
                                                                 target_weight_q * mask_s,
                                                                 target_sizes,
                                                                 reconstructed_keypoints,
                                                                 )
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head_module.get_accuracy(output[-1],
                                                                       target_keypoints,
                                                                       target_weight_q * mask_s,
                                                                       target_sizes,
                                                                       height=h)
            losses.update(keypoint_accuracy)
        return losses

    def forward_test(self,
                     img_s,
                     target_s,
                     target_weight_s,
                     img_q,
                     target_q,
                     target_weight_q,
                     img_metas=None,
                     vis_offset=True,
                     **kwargs):

        """Defines the computation performed at every call when testing."""
        batch_size, _, img_height, img_width = img_q.shape
        output, initial_proposals, similarity_map, mask_s, reconstructed_keypoints = self.predict(img_s,
                                                                                                  target_s,
                                                                                                  target_weight_s,
                                                                                                  img_q,
                                                                                                  img_metas
                                                                                                  )
        predicted_pose = output[-1].detach().cpu().numpy()
        result = {}

        if self.with_keypoint:
            keypoint_result = self.keypoint_head_module.decode(img_metas, predicted_pose, img_size=[img_width, img_height])
            result.update(keypoint_result)

        if vis_offset:
            result.update({"points": torch.cat((initial_proposals[None], output)).cpu().numpy()})

        result.update({"sample_image_file": [img_metas[i]['sample_image_file'] for i in range(len(img_metas))]})

        return result

    def predict(self,
                img_s,
                target_s,
                target_weight_s,
                img_q,
                img_metas=None,
                random_mask=None):

        batch_size, _, img_height, img_width = img_q.shape
        assert [i['sample_skeleton'][0] != i['query_skeleton'] for i in img_metas]
        mask_s = target_weight_s[0]
        for target_weight in target_weight_s:
            mask_s = mask_s * target_weight
        feature_q, feature_s = self.extract_features(img_s, img_q)
        skeleton_lst = [i['sample_skeleton'][0] for i in img_metas]

        (output, initial_proposals, similarity_map, reconstructed_keypoints) = self.keypoint_head_module(
            feature_q, feature_s, target_s, mask_s, skeleton_lst, random_mask=random_mask)

        return output, initial_proposals, similarity_map, mask_s, reconstructed_keypoints

    def extract_features(self, img_s, img_q):
        with torch.no_grad():
            dino_feature_s = [self.encoder_sample(img) for img in img_s]
            dino_feature_q = self.encoder_query(img_q)  # [bs, 3, h, w]
        if self.freeze_backbone:
            with torch.no_grad():
                feature_s = [self.probe(f) for f in dino_feature_s]
                feature_q = self.probe(dino_feature_q)
        else:
            feature_s = [self.probe(f) for f in dino_feature_s]
            feature_q = self.probe(dino_feature_q)

        return feature_q, feature_s

    def parse_keypoints_from_img_meta(self, img_meta, device, keyword='query'):
        """Parse keypoints from the img_meta.

        Args:
            img_meta (dict): Image meta info.
            device (torch.device): Device of the output keypoints.
            keyword (str): 'query' or 'sample'. Default: 'query'.

        Returns:
            Tensor: Keypoints coordinates of query images.
        """

        if keyword == 'query':
            query_kpt = torch.stack([
                torch.tensor(info[f'{keyword}_joints_3d']).to(device) for info in img_meta], dim=0)[:, :, :2]
        else:
            query_kpt = []
            for info in img_meta:
                if isinstance(info[f'{keyword}_joints_3d'][0], torch.Tensor):
                    samples = torch.stack(info[f'{keyword}_joints_3d'])
                else:
                    samples = np.array(info[f'{keyword}_joints_3d'])
                query_kpt.append(torch.tensor(samples).to(device)[:, :, :2])
            query_kpt = torch.stack(query_kpt, dim=0)  # [bs, , num_samples, num_query, 2]
        return query_kpt

    def get_full_similarity_map(self, feature_q, feature_s, h, w):
        resized_feature_q = F.interpolate(feature_q, size=(h, w),
                                          mode='bilinear')
        resized_feature_s = [F.interpolate(s, size=(h, w), mode='bilinear') for
                             s in feature_s]
        return [self.chunk_cosine_sim(f_s, resized_feature_q) for f_s in
                resized_feature_s]

    # UNMODIFIED
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_limb_color=None,
                    radius=4,
                    text_color=(255, 0, 0),
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_limb_color (np.array[Mx3]): Color of M limbs.
                If None, do not draw limbs.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """

        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        bbox_result = []
        pose_result = []
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            mmcv.imshow_bboxes(
                img,
                bboxes,
                colors=bbox_color,
                top_k=-1,
                thickness=thickness,
                show=False,
                win_name=win_name,
                wait_time=wait_time,
                out_file=None)

            for person_id, kpts in enumerate(pose_result):
                # draw each point on image
                if pose_kpt_color is not None:
                    assert len(pose_kpt_color) == len(kpts), (
                        len(pose_kpt_color), len(kpts))
                    for kid, kpt in enumerate(kpts):
                        x_coord, y_coord, kpt_score = int(kpt[0]), int(
                            kpt[1]), kpt[2]
                        if kpt_score > kpt_score_thr:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)
                            transparency = max(0, min(1, kpt_score))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

                # draw limbs
                if skeleton is not None and pose_limb_color is not None:
                    assert len(pose_limb_color) == len(skeleton)
                    for sk_id, sk in enumerate(skeleton):
                        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                        1]))
                        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                        1]))
                        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                and pos1[1] < img_h and pos2[0] > 0
                                and pos2[0] < img_w and pos2[1] > 0
                                and pos2[1] < img_h
                                and kpts[sk[0] - 1, 2] > kpt_score_thr
                                and kpts[sk[1] - 1, 2] > kpt_score_thr):
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1]) ** 2 + (
                                    X[0] - X[1]) ** 2) ** 0.5
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)), int(angle),
                                0, 360, 1)

                            r, g, b = pose_limb_color[sk_id]
                            cv2.fillConvexPoly(img_copy, polygon,
                                               (int(r), int(g), int(b)))
                            transparency = max(
                                0,
                                min(
                                    1, 0.5 *
                                       (kpts[sk[0] - 1, 2] + kpts[
                                           sk[1] - 1, 2])))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

        show, wait_time = 1, 1
        if show:
            height, width = img.shape[:2]
            max_ = max(height, width)

            factor = min(1, 800 / max_)
            enlarge = cv2.resize(
                img, (0, 0),
                fx=factor,
                fy=factor,
                interpolation=cv2.INTER_CUBIC)
            imshow(enlarge, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
