import torch
import torch.nn as nn
from torch.nn import functional as F

import clip
from common.nets.attn import ViT
import numpy as np
import math
import main.config as cfg
def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

class TAI(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # self.device = torch.device(0)
        self.prior_prob = cfg.TAI_prior_prob

        self.num_keypoints = cfg.h_num_keypoints
        self.in_channels = cfg.TAI_in_channels
        self.channels = cfg.TAI_channels
        self.out_channels = cfg.TAI_out_channels
        assert self.out_channels == self.num_keypoints

        self.dim_redu = cfg.h_dim_reduction
        self.global_text_channels = cfg.h_global_text_channels

        self.joints_text_channels = cfg.joints_text_channels
        if self.dim_redu:
            self.inter_channels = cfg.TAI_inter_channels
        else:
            self.inter_channels = self.global_text_channels

        self.c_attn = ChannelAtten(self.in_channels, self.channels)
        self.s_attn = SpatialAtten(self.in_channels, self.channels)
        self.fuse_attn = nn.Conv2d(self.channels * 2, self.channels, 1, 1, 0,bias = True)
        self.heatmap_conv = nn.Conv2d(self.channels, self.out_channels, 1, 1, 0)
        # self.prior_prob = 0.01
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.heatmap_conv.bias.data.fill_(bias_value)

        self.mse_loss = nn.MSELoss(reduce = True)
        self.ce_loss = nn.CrossEntropyLoss(reduce = True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)


        self.vit_global = ViT(
            dim=128,
            image_size=32,
            patch_size=4,
            heads=8,
            dim_head=16,
            mlp_dim=64,
            channels=32,
        )
        self.vit = ViT(
            dim=128,
            image_size=32,
            patch_size=4,
            heads=8,
            dim_head=16,
            mlp_dim=64,
            channels=32,
        )
        self.joint_text_features = None
        self.inter_channels = cfg.h_inter_channels # 128
        self.conv_down = nn.Conv2d(self.in_channels, self.channels,1, 1, 0,bias = True)
        self.logit_scale = 1.0
        self.fc_inst_em = nn.Linear(self.channels, self.inter_channels,bias=True)
        self.fc_inst_text = nn.Linear(self.global_text_channels, self.inter_channels,bias=True)
        self.fc_pixel_text = nn.Linear(self.global_text_channels, self.inter_channels)
        self.fc_joint_img = nn.Linear(self.channels, self.inter_channels)
        self.fc_joint_text = nn.Linear(self.channels, self.inter_channels)
        self.ins_cov_down = nn.Conv2d(self.channels, 1, 1, 1, 0,bias = True)
        self.decoder_layer_ins = nn.TransformerDecoderLayer(d_model=self.inter_channels, nhead=8, dropout=0.1,batch_first=True)
        self.transformer_decoder_ins = nn.TransformerDecoder(self.decoder_layer_ins, num_layers=1)

        self.decoder_layer_joint = nn.TransformerDecoderLayer(d_model=self.inter_channels, nhead=8, dropout=0.1,batch_first=True)
        self.transformer_decoder_joint = nn.TransformerDecoder(self.decoder_layer_joint, num_layers=1)

        self.clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False,download_root='/home/zzh23/codes/gcn32z/CLIP-main')

        self.category_dict = {
            'object': ['master_chef_can', 'cracker_box', 'sugar_box''tomato_soup_can', 'mustard _bottle',
                       'tuna_fish_can', 'pudding _box', 'gelatin_box', 'potted_meat_can', 'banana',
                       'pitcher_base', 'bleach_cleanser', 'bowl', 'mug', 'power_drill', 'wood_block',
                       'scissors', 'large_marker', 'extra_large_clamp', 'foam_brick'],
            'finger': ['Wrist',
                       'MCP of the thumb','PIP of the thumb', 'DIP of the thumb', 'TIP of the thumb',
                       'MCP of the index finger','PIP of the index finger','DIP of the index finger','TIP of the index finger',
                       'MCP of the middle finger', 'PIP of the middle finger','DIP of the middle finger','TIP of the middle finger',
                       'MCP of the ring finger','PIP of the ring finger','DIP of the ring finger', 'TIP of the ring finger',
                       'MCP of the pinky', 'PIP of the pinky','DIP of the pinky', 'TIP of the pinky']
            # TIP，DIP，PIP，CMC，MCP
        }

    def forward(self,feats,inputs):
        b,c,h,w = feats.size()
        feats = feats.reshape(b, -1,cfg.TAI_shape,cfg.TAI_shape).cuda()
        global_features = self.conv_down(feats)
        instance_features = global_features.cuda()
        instance_params = []
        for i in range(feats.size(0)):
            instance_params.append(self._sample_feats(feats[i], inputs['joints_img_centers'][i]).reshape(1,-1))
        instance_params = torch.cat(instance_params,dim=0)
        c_instance_feats = self.c_attn(instance_features, instance_params)
        s_instance_feats = self.s_attn(instance_features, instance_params, inputs['joints_img_centers'])
        cond_instance_feats = torch.cat((c_instance_feats, s_instance_feats),dim=1)
        cond_instance_feats = self.fuse_attn(cond_instance_feats)
        cond_instance_feats = F.relu(cond_instance_feats)


        if self.training:
            inst_img_feats_norm = self.img_norm_global(instance_features)
            joint_img_feats_norm = self.img_norm(cond_instance_feats)

            global_loss = self.global_loss(inst_img_feats_norm,inputs,cond_instance_feats)

            joints_loss, pixel_loss = self.joint_loss(joint_img_feats_norm, inputs)
            return  global_loss,joints_loss,pixel_loss


    def global_loss(self, img_features, inputs,cond_instance_feats):
        b, c, h, w = img_features.size()
        img_features_ = self.fc_inst_em(img_features.permute(0, 2, 3, 1))
        input_temp = inputs.copy()
        ins_text_feats = input_temp['text_g_features'].reshape(b,-1)
        # text_list = np.array(input_temp['global_text']).squeeze()
        # # text_list = np.array(input_temp['focal_text']).squeeze()
        # text_list =text_list.tolist()

        # with torch.no_grad():
        #     # ins_loc_text = clip.tokenize(text_list)
        #     # ins_loc_text = clip.tokenize(text_list) # 64,77
        #     # ins_text_feats = self.clip_pretrained.encode_text(ins_loc_text.cuda()) # 64,512
        #     #longclip切换
        #     ins_loc_text = longclip.tokenize(text_list) #(32,248)
        #     ins_text_feats = self.longclip_pretrained.encode_text(ins_loc_text.cuda()) #(32,768)

        if self.dim_redu:
            inst_text_features_norm = self.fc_inst_text(ins_text_feats.float())
        else:
            inst_text_features_norm = ins_text_feats.float()

        tgt = img_features_.reshape(b, h * w, -1)
        inst_text_features = self.transformer_decoder_ins(inst_text_features_norm.unsqueeze(1), tgt.detach())
        inst_text_features = inst_text_features / (inst_text_features.norm(dim=-1, keepdim=True) + 1e-5)
        logits_per_image = (self.logit_scale * tgt @ inst_text_features.permute(0, 2, 1)).squeeze()
        out_features = logits_per_image.reshape(b, h, w)
        ins_img_feat = self.ins_cov_down(cond_instance_feats).reshape(b, h, w)
        ins_img_feat = ins_img_feat / (ins_img_feat.norm(dim=(1, 2), keepdim=True) + 1e-5)
        global_loss = F.mse_loss(out_features, ins_img_feat,reduce = True)

        return global_loss
    def joint_loss(self,image_features_norm, inputs):
        b, c, h, w = image_features_norm.size()
        gt_instance_heatmaps = inputs['inst_heatmaps']
        gt_instance_masks = inputs['inst_masks']
        hm_mask = gt_instance_heatmaps.unsqueeze(2) * gt_instance_masks.unsqueeze(2)
        joint_features = image_features_norm.unsqueeze(1) * hm_mask
        hm_mask = hm_mask.reshape(b, self.num_keypoints, 1, -1)
        image_features_sem = joint_features.reshape(b, self.num_keypoints, c, -1).sum(-1) / (hm_mask.sum(-1) + 1e-5)
        image_features_sem = self.fc_joint_img(image_features_sem)
        if self.joint_text_features is None:
            with torch.no_grad():
                joint_text = clip.tokenize(self.category_dict['finger'])
                self.joint_text_features = self.clip_pretrained.encode_text(joint_text.cuda().detach())
        if self.dim_redu:
            joint_text_features =  self.fc_pixel_text(self.joint_text_features.float())
        else:
            joint_text_features =  self.joint_text_features.float()
        tgt = self.fc_joint_text(image_features_norm.reshape(b, c, h*w).permute(0, 2, 1))
        joint_text_features = self.transformer_decoder_joint(joint_text_features.unsqueeze(0).expand([b, -1, -1]),tgt.detach())
        joint_text_features = joint_text_features / (joint_text_features.norm(dim=-1, keepdim=True) + 1e-5)
        logits_per_image = (self.logit_scale * tgt @ joint_text_features.permute(0, 2, 1))
        out_features = logits_per_image.permute(0, 2, 1).reshape(b, -1, h, w)
        pixel_heatmap_loss = self.mse_loss(out_features, gt_instance_heatmaps)
        similarities = torch.matmul(self.logit_scale * image_features_sem, joint_text_features.permute(0, 2, 1))
        labels = torch.arange(self.num_keypoints)
        labels = labels.unsqueeze(0).expand([b, -1]).cuda()
        joint_semantic_loss = (self.ce_loss(similarities, labels) + self.ce_loss(similarities.permute(0, 2, 1),labels)) / 2

        return joint_semantic_loss, pixel_heatmap_loss
    def img_norm(self, input_img):
        img_feat_att = self.vit(input_img)
        img_feats_norm = img_feat_att / (img_feat_att.norm(dim=(2,3), keepdim=True) + 1e-5)
        return img_feats_norm

    def img_norm_global(self, input_img):
        img_feat_att = self.vit_global(input_img)
        img_feats_norm = img_feat_att / (img_feat_att.norm(dim=(2,3), keepdim=True) + 1e-5)
        return img_feats_norm

    def _sample_feats(self, features, pos_ind):
        pos_ind  = torch.div(pos_ind, 32, rounding_mode='floor').to(pos_ind.dtype)
        feats = features[:, pos_ind[:, 0], pos_ind[:, 1]]
        return feats.permute(1, 0)

class ChannelAtten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAtten, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels,bias=True)

    def forward(self, global_features, instance_params):
        B, C, H, W = global_features.size() #B, C, H, W =64, 32, 32, 32
        instance_params = self.atn(instance_params)
        instance_params = instance_params.reshape(B, C, 1, 1)
        return global_features * instance_params.expand_as(global_features)

class SpatialAtten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAtten, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels,bias=True)
        self.feat_stride = 4
        conv_in = 3
        self.conv = nn.Conv2d(conv_in, 1, 5, 1, 2,bias=True)

    def forward(self, global_features, instance_params, instance_inds):
        B, C, H, W = global_features.size()
        instance_params = self.atn(instance_params)
        instance_params = instance_params.reshape(B, C, 1, 1)
        feats = global_features * instance_params.expand_as(global_features)
        fsum = torch.sum(feats, dim=1, keepdim=True)
        input_feats = fsum #
        locations = compute_locations(global_features.size(2), global_features.size(3), stride=1, device=global_features.device)
        n_inst = instance_inds.size(0)
        H, W = global_features.size()[2:]
        instance_locations = torch.flip(instance_inds, [1])
        instance_locations = instance_locations#
        relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coords = relative_coords.permute(0, 2, 1).float()
        relative_coords = (relative_coords / 32).to(dtype=global_features.dtype)
        relative_coords = relative_coords.reshape(n_inst, 2, H, W)
        input_feats = torch.cat((input_feats, relative_coords), dim=1)
        mask = self.conv(input_feats).sigmoid()
        return global_features * mask

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2

    return locations



