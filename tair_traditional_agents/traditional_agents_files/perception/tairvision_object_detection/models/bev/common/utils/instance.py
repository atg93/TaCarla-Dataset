import torch
import numpy as np

from tairvision.models.bev.common.utils.geometry import (mat2pose_vec, pose_vec2mat, warp_features,
                                                         cumulative_warp_features_reverse)


# Set ignore index to 0 for vis
def convert_instance_mask_to_center_and_offset_label(instance_img, future_egomotion, num_instances, ignore_index=255,
                                                     subtract_egomotion=True, sigma=3, spatial_extent=None):
    seq_len, _, h, w = instance_img.shape
    center_label = torch.zeros(seq_len, 1, h, w)
    offset_label = ignore_index * torch.ones(seq_len, 2, h, w)
    future_displacement_label = ignore_index * torch.ones(seq_len, 2, h, w)
    # x is vertical displacement, y is horizontal displacement
    x, y = torch.meshgrid(torch.arange(h, dtype=torch.float), torch.arange(w, dtype=torch.float))

    if subtract_egomotion:
        future_egomotion_inv = mat2pose_vec(pose_vec2mat(future_egomotion).inverse())

    # Compute warped instance segmentation
    warped_instance_seg = {}
    for t in range(1, seq_len):
        warped_inst_t = warp_features(instance_img[t].unsqueeze(0).float(),
                                      future_egomotion_inv[t - 1].unsqueeze(0), mode='nearest',
                                      spatial_extent=spatial_extent)
        warped_instance_seg[t] = warped_inst_t[0, 0]

    # Ignore id 0 which is the background
    for instance_id in range(1, num_instances+1):
        prev_xc = None
        prev_yc = None
        prev_mask = None
        for t in range(seq_len):
            instance_mask = (instance_img[t, 0] == instance_id)
            if instance_mask.sum() == 0:
                # This instance is not in this frame
                prev_xc = None
                prev_yc = None
                prev_mask = None
                continue

            xc = x[instance_mask].mean().round().long()
            yc = y[instance_mask].mean().round().long()

            off_x = xc - x
            off_y = yc - y
            g = torch.exp(-(off_x ** 2 + off_y ** 2) / sigma ** 2)
            center_label[t, 0] = torch.maximum(center_label[t, 0], g)
            offset_label[t, 0, instance_mask] = off_x[instance_mask]
            offset_label[t, 1, instance_mask] = off_y[instance_mask]

            if prev_xc is not None:
                # Old method
                # cur_pt = torch.stack((xc, yc)).unsqueeze(0).float()
                # if subtract_egomotion:
                #     cur_pt = warp_points(cur_pt, future_egomotion_inv[t - 1])
                # cur_pt = cur_pt.squeeze(0)

                warped_instance_mask = warped_instance_seg[t] == instance_id
                if warped_instance_mask.sum() > 0:
                    warped_xc = x[warped_instance_mask].mean().round()
                    warped_yc = y[warped_instance_mask].mean().round()

                    delta_x = warped_xc - prev_xc
                    delta_y = warped_yc - prev_yc
                    future_displacement_label[t - 1, 0, prev_mask] = delta_x
                    future_displacement_label[t - 1, 1, prev_mask] = delta_y

            prev_xc = xc
            prev_yc = yc
            prev_mask = instance_mask

    return center_label, offset_label, future_displacement_label


# Set ignore index to 0 for vis
def instance_to_center_offset_flow(instance_img, future_egomotion, ignore_index=255,
                                   subtract_egomotion=True, sigma=3, spatial_extent=None):
    batch_len, seq_len, _, h, w = instance_img.shape
    center_label = torch.zeros(batch_len, seq_len, 1, h, w, device=instance_img.device)
    offset_label = ignore_index * torch.ones(batch_len, seq_len, 2, h, w, device=instance_img.device)
    future_displacement_label = ignore_index * torch.ones(batch_len, seq_len, 2, h, w, device=instance_img.device)
    # x is vertical displacement, y is horizontal displacement
    x, y = torch.meshgrid(torch.arange(h, dtype=torch.float, device=instance_img.device),
                          torch.arange(w, dtype=torch.float, device=instance_img.device))

    for b in range(batch_len):
        if subtract_egomotion:
            future_egomotion_inv = future_egomotion[b].inverse()

        # Compute warped instance segmentation
        warped_instance_seg = {}
        for t in range(1, seq_len):
            warped_inst_t = warp_features(instance_img[b, t].unsqueeze(0).float(),
                                          future_egomotion_inv[t - 1].unsqueeze(0), mode='nearest',
                                          spatial_extent=spatial_extent)
            warped_instance_seg[t] = warped_inst_t[0, 0]

        num_instances = instance_img[b].max().item()
        # Ignore id 0 which is the background
        for instance_id in range(1, num_instances+1):
            prev_xc = None
            prev_yc = None
            prev_mask = None
            for t in range(seq_len):
                instance_mask = (instance_img[b, t, 0] == instance_id)
                if instance_mask.sum() == 0:
                    # This instance is not in this frame
                    prev_xc = None
                    prev_yc = None
                    prev_mask = None
                    continue

                xc = x[instance_mask].mean().round().long()
                yc = y[instance_mask].mean().round().long()

                off_x = xc - x
                off_y = yc - y
                g = torch.exp(-(off_x ** 2 + off_y ** 2) / sigma ** 2)
                center_label[b, t, 0] = torch.maximum(center_label[b, t, 0], g)
                offset_label[b, t, 0, instance_mask] = off_x[instance_mask]
                offset_label[b, t, 1, instance_mask] = off_y[instance_mask]

                if prev_xc is not None:
                    # Old method
                    # cur_pt = torch.stack((xc, yc)).unsqueeze(0).float()
                    # if subtract_egomotion:
                    #     cur_pt = warp_points(cur_pt, future_egomotion_inv[t - 1])
                    # cur_pt = cur_pt.squeeze(0)

                    warped_instance_mask = warped_instance_seg[t] == instance_id
                    if warped_instance_mask.sum() > 0:
                        warped_xc = x[warped_instance_mask].mean().round()
                        warped_yc = y[warped_instance_mask].mean().round()

                        delta_x = warped_xc - prev_xc
                        delta_y = warped_yc - prev_yc
                        future_displacement_label[b, t - 1, 0, prev_mask] = delta_x
                        future_displacement_label[b, t - 1, 1, prev_mask] = delta_y

                prev_xc = xc
                prev_yc = yc
                prev_mask = instance_mask

    return center_label, offset_label, future_displacement_label


def balance_zpos_targets(zpos, offset, ignore_index):
    ignore_mask = (offset == ignore_index)[:, :, 0:1]
    # Adding ignore mask for background z values.
    no_ignore_mask = (offset != ignore_index)[:, :, 0:1]
    fg_size = no_ignore_mask.sum()
    bg_size = fg_size * 5
    bg_ind = torch.where(ignore_mask.reshape(-1))[0]
    chosen_bg_ind = bg_ind[np.random.permutation(bg_ind.shape[0])][0:bg_size]
    b, s, c, h, w = ignore_mask.shape
    ignore_mask = ignore_mask.reshape(-1)
    ignore_mask[chosen_bg_ind] = False
    ignore_mask = ignore_mask.reshape(b, s, c, h, w).repeat(1, 1, 2, 1, 1)

    masked_zpos = zpos.clone()
    masked_zpos[ignore_mask] = ignore_index

    return masked_zpos


def get_targets_dynamic(batch, receptive_field=1, spatial_extent=(50., 50.), ignore_index=255):
    labels = {}
    time_index = receptive_field - 1

    segmentation = batch['segmentation'][:, time_index:].float()
    instance = batch['instance'][:, time_index:].float()
    future_egomotion = batch['future_egomotion'][:, time_index:]
    center, offset, flow = instance_to_center_offset_flow(batch['instance'],
                                                          batch['future_egomotion'],
                                                          ignore_index=ignore_index,
                                                          subtract_egomotion=True,
                                                          spatial_extent=spatial_extent)

    z_position = batch['z_position']
    z_position = balance_zpos_targets(z_position, offset, ignore_index)

    # Warp labels to present's reference frame
    segmentation = cumulative_warp_features_reverse(segmentation, future_egomotion, spatial_extent=spatial_extent)
    labels['segmentation'] = segmentation.long().contiguous()

    # Warp instance labels to present's reference frame
    instance = cumulative_warp_features_reverse(instance, future_egomotion, spatial_extent=spatial_extent)
    labels['instance'] = instance.long().contiguous()

    center = cumulative_warp_features_reverse(center[:, time_index:], future_egomotion, spatial_extent=spatial_extent)
    labels['centerness'] = center.contiguous()

    offset = cumulative_warp_features_reverse(offset[:, time_index:], future_egomotion, spatial_extent=spatial_extent)
    labels['offset'] = offset.contiguous()

    z_position = cumulative_warp_features_reverse(z_position[:, time_index:], future_egomotion,
                                                  spatial_extent=spatial_extent)
    labels['z_position'] = z_position.contiguous()

    flow = cumulative_warp_features_reverse(flow[:, time_index:], future_egomotion, spatial_extent=spatial_extent)
    labels['flow'] = flow.contiguous()

    return labels
