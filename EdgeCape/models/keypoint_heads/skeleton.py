import random
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmpose.models import HEADS
from EdgeCape.models.keypoint_heads.encoder_decoder import (TransformerDecoderLayer, _get_clones)


@HEADS.register_module()
class SkeletonPredictor(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_layers=3,
                 dim_feedforward=768,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 learn_skeleton: bool = False,
                 max_hop: int = 5,
                 adj_normalization: bool = True,
                 markov_bias: bool = True,
                 mask_res: bool = False,
                 use_zero_conv: bool = True,
                 max_hops: int = 4,
                 two_way_attn: bool = True,
                 gcn_norm: bool = False, ):
        super(SkeletonPredictor, self).__init__()
        if num_layers > 0:
            decoder_layer = TransformerDecoderLayer(d_model=d_model,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout,
                                                    activation='relu',
                                                    normalize_before=normalize_before,
                                                    max_hops=max_hops,
                                                    two_way_attn=two_way_attn)
            self.skeleton_predictor = _get_clones(decoder_layer, num_layers)
        self.gcn_norm = gcn_norm
        self.image_project = nn.Conv2d(dim_feedforward, d_model, kernel_size=1)
        self.learn_skeleton = learn_skeleton
        self.max_hop = max_hop
        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Sigmoid()
        self.adj_normalization = adj_normalization
        self.markov_bias = markov_bias
        self.k_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.mh_linear = nn.Conv2d(nhead, 1, kernel_size=1)
        self.num_heads = nhead
        self.mask_res = mask_res
        self.use_zero_conv = use_zero_conv
        if self.use_zero_conv:
            self.zero_conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self,
                skeleton: list,
                kp_features: torch.Tensor,
                image_features: torch.Tensor,
                kp_mask: torch.Tensor,
                query_image_pos_embed: torch.Tensor,
                ) -> [torch.Tensor, torch.Tensor]:

        assert skeleton is not None
        b, num_pts, _ = kp_features.shape
        gt_adj, _ = self.adj_mx_from_edges(num_pts=num_pts,
                                           skeleton=skeleton,
                                           mask=kp_mask,
                                           device=kp_features.device)
        binary_adj = gt_adj[:, 1] > 0
        if not self.learn_skeleton:
            return gt_adj, None, binary_adj
        adj, adj_for_attn, unnnormalized_adj = self.predict_adj(image_features=image_features,
                                                                kp_features=kp_features,
                                                                kp_mask=kp_mask,
                                                                query_image_pos_embed=query_image_pos_embed,
                                                                gt_adj=binary_adj)
        return adj, adj_for_attn, unnnormalized_adj

    def refine_features(self,
                        image_features: torch.Tensor,
                        kp_features: torch.Tensor,
                        kp_mask: torch.Tensor,
                        query_image_pos_embed: torch.Tensor,
                        adj: torch.Tensor = None,
                        ):

        bs, num_pts, _ = kp_features.shape
        adj = self.soft_normalize_adj(adj, kp_mask)
        image_features = [self.image_project(image_feature) for image_feature in image_features]
        zero_pos_embed = torch.zeros_like(kp_features).flatten(2).permute(1, 0, 2)
        query_image_pos_embed = query_image_pos_embed.flatten(2).permute(2, 0, 1)
        concat_pos_embed = torch.cat((query_image_pos_embed, zero_pos_embed))
        kp_features = kp_features.flatten(2).permute(1, 0, 2)
        image_features = [image_feature.flatten(2).permute(2, 0, 1) for image_feature in image_features]
        tgt_key_padding_mask_remove_all_true = kp_mask.clone().to(kp_mask.device)
        tgt_key_padding_mask_remove_all_true[kp_mask.logical_not().sum(dim=-1) == 0, 0] = False
        kp_feat_lst = []
        for s, image_feature in enumerate(image_features):
            s_kp_features = kp_features.clone()
            for i, layer in enumerate(self.skeleton_predictor):
                s_kp_features, image_feature, _, _, _ = layer(
                    s_kp_features,
                    image_feature,
                    tgt_key_padding_mask=tgt_key_padding_mask_remove_all_true,
                    concat_pos_embed=concat_pos_embed,
                    init_pos_emb=zero_pos_embed,
                    adj=adj,
                )
            kp_feat_lst.append(s_kp_features.permute(1, 0, 2))
        kp_features = torch.mean(torch.stack(kp_feat_lst, dim=0), 0)

        return kp_features

    def predict_adj(self,
                    image_features: torch.Tensor,
                    kp_features: torch.Tensor,
                    kp_mask: torch.Tensor,
                    query_image_pos_embed: torch.Tensor,
                    gt_adj: torch.Tensor = None):

        kp_features = self.refine_features(image_features,
                                           kp_features,
                                           kp_mask,
                                           query_image_pos_embed,
                                           gt_adj)

        normalized_adj, unnnormalized_adj = self.predict_skeleton(kp_features, kp_mask, gt_adj)
        attn_bias_matrix = self.markov_transition_matrix(normalized_adj[:, 1])
        return normalized_adj, attn_bias_matrix, unnnormalized_adj

    def predict_skeleton(self, kp_features, kp_mask, gt_adj):
        bs, num_pts, _ = kp_features.shape
        # Self-attention matrix from kp_features
        kp_features = kp_features.permute(1, 0, 2) * ~kp_mask.transpose(0, 1).unsqueeze(-1)
        q_kp = self.q_proj(kp_features).contiguous().view(num_pts, bs * self.num_heads, -1).transpose(0, 1)
        k_kp = self.k_proj(kp_features).contiguous().view(num_pts, bs * self.num_heads, -1).transpose(0, 1)
        attn = torch.bmm(q_kp, k_kp.transpose(1, 2)).view(bs, self.num_heads, num_pts, num_pts)
        unnormalized_adj_matrix = self.mh_linear(attn).squeeze(1)
        unnormalized_adj_matrix = (unnormalized_adj_matrix + unnormalized_adj_matrix.transpose(1, 2)) / 2
        unnormalized_adj_matrix = self.combine_adj(gt_adj, unnormalized_adj_matrix)
        unnormalized_adj_matrix = self.activation(unnormalized_adj_matrix)
        normalized_adj = self.soft_normalize_adj(unnormalized_adj_matrix, kp_mask, gt_adj)
        unnormalized_adj_matrix = unnormalized_adj_matrix * ~kp_mask.unsqueeze(-1) * ~kp_mask.unsqueeze(-2)
        return normalized_adj, unnormalized_adj_matrix

    def combine_adj(self, gt_adj, predicted_adj):
        if self.use_zero_conv:
            predicted_adj = self.zero_conv(predicted_adj.unsqueeze(1)).squeeze(1)
        adj = gt_adj + predicted_adj
        return adj

    def markov_transition_matrix(self, adj):
        """
        Compute the Markov transition matrix from the adjacency matrix.
        :param adj: (bs, num_pts, num_pts)
        :return: (bs, num_pts, num_pts)
        """
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
        transfer_mat = [torch.matrix_power(adj.float(), d) for d in range(self.max_hop + 1)]
        arrive_mat = torch.stack(transfer_mat)
        return arrive_mat

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        """Initialize weights of the transformer head."""
        # nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
        # nn.init.constant_(self.input_proj.bias, 0)

    def adj_mx_from_edges(self, num_pts, skeleton, mask=None, device='cuda'):
        binary_adj_mx = torch.empty(0, device=device)
        batch_size = len(skeleton)
        for b in range(batch_size):
            edges = torch.tensor(skeleton[b])
            adj = torch.zeros(num_pts, num_pts, device=device)
            if len(edges.shape) > 1:
                adj[edges[:, 0], edges[:, 1]] = 1
                adj[edges[:, 1], edges[:, 0]] = 1
            binary_adj_mx = torch.concatenate((binary_adj_mx, adj.unsqueeze(0)), dim=0)
        if mask is not None:
            adj = self.normalize_adj(binary_adj_mx, mask)
        else:
            adj = None
        return adj, binary_adj_mx

    def normalize_adj(self, binary_adj_mx, mask):
        trans_adj_mx = torch.transpose(binary_adj_mx, 1, 2)
        cond = (trans_adj_mx > binary_adj_mx).float()
        adj_unnormalized = binary_adj_mx + trans_adj_mx * cond - binary_adj_mx * cond
        adj = adj_unnormalized * ~mask[..., None] * ~mask[:, None]
        adj = torch.nan_to_num(adj / adj.sum(dim=-1, keepdim=True))
        adj = torch.stack((torch.diag_embed(~mask), adj), dim=1)
        return adj

    def soft_normalize_adj(self, adj_mx, mask, gt_adj=None):
        adj_mask = ~mask[..., None] * ~mask[:, None]
        if self.mask_res and gt_adj is not None:
            adj_mask = gt_adj
        adj = adj_mx * adj_mask
        if self.adj_normalization:
            adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
        if not self.gcn_norm:
            adj = torch.stack((torch.diag_embed(~mask), adj), dim=1)
        return adj
