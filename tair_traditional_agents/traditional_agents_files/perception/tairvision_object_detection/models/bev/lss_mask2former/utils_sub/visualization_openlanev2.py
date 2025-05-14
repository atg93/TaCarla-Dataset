from tairvision.models.bev.lss.utils.visualization import INSTANCE_COLOURS
import numpy as np
from utils.openlanev2_visualization import draw_annotation_pv_centerline
import torch
from tairvision.models.bev.common.openlanev2.process import decide_mean_std_v2
from tairvision.models.bev.lss_mask2former.utils_sub.evaluate_openlaneV2_functions import comb


class VisualizationModuleOpenLaneV2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bev_size = (
            int((cfg.LIFT.X_BOUND[1] - cfg.LIFT.X_BOUND[0]) / cfg.LIFT.X_BOUND[2]),
            int((cfg.LIFT.Y_BOUND[1] - cfg.LIFT.Y_BOUND[0]) / cfg.LIFT.Y_BOUND[2])
        )

        self.mean, self.std = decide_mean_std_v2(cfg)

    def get_bitmap(self, x, bev_size=(200, 200)):
        output = 255 * np.ones((*bev_size, 3)).astype(np.uint8)

        x_mask = (x > 0) * (x < 70)

        colors = INSTANCE_COLOURS[x[x_mask]]
        output[x_mask] = colors  # [255, 172, 28]
        center_x = bev_size[0] // 2
        center_y = bev_size[1] // 2
        output[center_x-5:center_x+5, center_y-3:center_y+3] = [52, 152, 219]
        output = self.pad_images(output, ones=False)

        return output
    
    def _visualize_images_with_centerlines(self, batch, targets, image_index_list):
        image_list = []
        batch_index = 0
        for image_index in image_index_list:
            selected_image = batch["images"][batch_index, 0, image_index, ...].permute(1, 2, 0)
            selected_image = selected_image.detach().cpu().numpy()
            selected_image = selected_image * self.std + self.mean
            selected_image = (selected_image * 255).astype(np.uint8)

            extrinsic = batch["cams_to_lidar"][batch_index][0][image_index].detach().cpu().numpy()
            intrinsic = batch["intrinsics"][batch_index][0][image_index].detach().cpu().numpy()
            ego_pose_dict = {"rotation": extrinsic[:3, :3], "translation": extrinsic[:3, 3]}
            intrinsic_dict = {"K": intrinsic}

            centerline_list = self.process_bezier_regressions(batch, targets, batch_index)

            center_line_selected_image = draw_annotation_pv_centerline(
                selected_image.copy(), 
                centerline_list,
                intrinsic=intrinsic_dict,
                extrinsic=ego_pose_dict,
                with_attribute=False
                )
            
            image_list.append(center_line_selected_image)
        center_line_merged_images = np.concatenate(image_list, axis=1)
        return center_line_merged_images
    
    def process_centerlines(self, batch, targets, batch_index):
        centerline_list = []
        for centerline in batch["centerlines_list"][batch_index][0]:
            centerline = centerline[0]
            centerline_concat = np.concatenate([centerline, np.ones((centerline.shape[0], 1))], axis=1)
            view_inv = targets["view_inv"][0, 0, 0].cpu().numpy()
            unprojected_centerlines = view_inv @ centerline_concat.T
            unprojected_centerlines = unprojected_centerlines.T[:, :3]
            centerline_dict = {"points": unprojected_centerlines}
            centerline_list.append(centerline_dict)
        return centerline_list
    
    def process_bezier_regressions(self, batch, targets, batch_index):
        centerline_list = []
        bezier_regressions = targets["dab"][batch_index]["regressions"]
        lanes = bezier_regressions.reshape(-1, bezier_regressions.shape[-1] // 3, 3)
    
        n_points = 11
        n_control = lanes.shape[1]
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        bezier_A = torch.tensor(A, dtype=torch.float32, device=lanes.device)
        lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
        lanes = lanes.cpu().numpy()
        
        for lane in lanes:
            lane_unprojected = np.concatenate([lane, np.ones((lane.shape[0], 1))], axis=1)
            view_inv = targets["view_inv"][0, 0, 0].cpu().numpy()
            unprojected_centerlines = view_inv @ lane_unprojected.T
            unprojected_centerlines = unprojected_centerlines.T[:, :3]
            lane_dict = {"points": unprojected_centerlines}
            centerline_list.append(lane_dict)

        return centerline_list

    
    def get_bev_maps(self, batch, post_output, targets):
        segm_target_raw = targets['segmentation'][0, 0, 0].cpu().numpy()
        segm_target = self.get_bitmap(segm_target_raw, bev_size=self.bev_size)

        inst_target_raw_1 = targets['instance'][0, 0, 0].cpu().numpy()
        inst_target_1 = self.get_bitmap(inst_target_raw_1, bev_size=self.bev_size)

        inst_target_raw_2 = targets['instance'][0, 0, 1].cpu().numpy()
        inst_target_2 = self.get_bitmap(inst_target_raw_2, bev_size=self.bev_size)

        inst_target_raw_3 = targets['instance'][0, 0, 2].cpu().numpy()
        inst_target_3 = self.get_bitmap(inst_target_raw_3, bev_size=self.bev_size)

        inst_target_raw_4 = targets['instance'][0, 0, 3].cpu().numpy()
        inst_target_4 = self.get_bitmap(inst_target_raw_4, bev_size=self.bev_size)
        
        segm_pred_raw = post_output['segm'][0, 0, 0].cpu().numpy()
        segm_pred = self.get_bitmap(segm_pred_raw, bev_size=self.bev_size)

        inst_pred_raw_1 = post_output['inst'][0, 0, 0].cpu().numpy()
        inst_pred_1 = self.get_bitmap(inst_pred_raw_1, bev_size=self.bev_size)

        inst_pred_raw_2 = post_output['inst'][0, 0, 1].cpu().numpy()
        inst_pred_2 = self.get_bitmap(inst_pred_raw_2, bev_size=self.bev_size)

        inst_pred_raw_3 = post_output['inst'][0, 0, 2].cpu().numpy()
        inst_pred_3 = self.get_bitmap(inst_pred_raw_3, bev_size=self.bev_size)

        inst_pred_raw_4 = post_output['inst'][0, 0, 3].cpu().numpy()
        inst_pred_4 = self.get_bitmap(inst_pred_raw_4, bev_size=self.bev_size)

        
        map_target = np.concatenate([
            segm_target, 
            inst_target_1,
            inst_target_2, 
            inst_target_3,
            inst_target_4,
        ], axis=1)

        map_pred = np.concatenate([
            segm_pred, 
            inst_pred_1,
            inst_pred_2, 
            inst_pred_3,
            inst_pred_4,
        ], axis=1)

        map = np.concatenate([map_target, map_pred], axis=0)

        return map, None, None
    
    @staticmethod
    def pad_images(input_image, padding_size=5, ones=False):
        h, w, c = input_image.shape

        padding_func = np.ones if ones else np.zeros
        vertical_padding = padding_func((padding_size, w, c)).astype(np.uint8) * 255
        horizontal_padding = padding_func((h + padding_size * 2, padding_size, c)).astype(np.uint8) * 255

        output_image = np.concatenate([input_image, vertical_padding], axis=0)
        output_image = np.concatenate([vertical_padding, output_image], axis=0)
        output_image = np.concatenate([output_image, horizontal_padding], axis=1)
        output_image = np.concatenate([horizontal_padding, output_image], axis=1)

        return output_image