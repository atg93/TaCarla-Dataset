import copy
import logging
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange

from transformers import (
    AutoConfig,
    AutoModel,
)

import cv2

logger = logging.getLogger(__name__)


class HFLM(nn.Module):
    def __init__(self, config_net, config_all):
        super().__init__()
        self.config_all = config_all
        self.config_net = config_net

        self.object_types = 2 + 1  # vehicles, route +1 for padding and wp embedding
        self.num_attributes = 6  # x,y,yaw,speed/id, extent x, extent y

        precisions = [
            self.config_all.model.pre_training.get("precision_pos", 4),
            self.config_all.model.pre_training.get("precision_pos", 4),
            self.config_all.model.pre_training.get("precision_angle", 4),
            self.config_all.model.pre_training.get("precision_speed", 4),
            self.config_all.model.pre_training.get("precision_pos", 4),
            self.config_all.model.pre_training.get("precision_pos", 4),
        ]

        self.vocab_size = [2 ** i for i in precisions]

        # model
        config = AutoConfig.from_pretrained(
            self.config_net.hf_checkpoint
        )  # load config from hugging face model
        n_embd = config.hidden_size
        self.model = AutoModel.from_config(config=config)

        # sequence padding for batching
        self.cls_emb = nn.Parameter(
            torch.randn(1, self.num_attributes + 1)
        )  # +1 because at this step we still have the type indicator
        self.eos_emb = nn.Parameter(
            torch.randn(1, self.num_attributes + 1)
        )  # unnecessary TODO: remove

        # token embedding
        self.tok_emb = nn.Linear(self.num_attributes, n_embd)
        # object type embedding
        self.obj_token = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, self.num_attributes))
                for _ in range(self.object_types)
            ]
        )
        self.obj_emb = nn.ModuleList(
            [nn.Linear(self.num_attributes, n_embd) for _ in range(self.object_types)]
        )
        self.drop = nn.Dropout(config_net.embd_pdrop)

        # decoder head forecasting
        if (
                self.config_all.model.pre_training.get("pretraining", "none") == "forecast"
                or self.config_all.model.training.get("pretraining_path", "none") != "none"
        ):
            # one head for each attribute type -> we have different precision per attribute
            self.heads = nn.ModuleList(
                [
                    nn.Linear(n_embd, self.vocab_size[i])
                    for i in range(self.num_attributes)
                ]
            )

        # wp (CLS) decoding
        self.wp_head = nn.Linear(n_embd, 512)
        self.wp_decoder = nn.GRUCell(input_size=4, hidden_size=1025)
        self.wp_relu = nn.ReLU()
        self.wp_output = nn.Linear(1025, 2)

        # other actor wp (CLS) decoding
        self.other_wp_head = nn.Linear(512, 512)
        self.other_wp_decoder = nn.GRUCell(input_size=2, hidden_size=512)
        self.other_wp_relu = nn.ReLU()
        self.other_wp_output = nn.Linear(512, 2)

        # PID controller
        self.turn_controller = PIDController(K_P=0.9, K_I=0.75, K_D=0.3, n=20)
        self.speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

        self.other_gru_wp_head_layer = nn.Linear(320, 512)
        self.task_name_layer = nn.Linear(512, 38)
        self.add_learnable_input = True

        # torch.rand(1, 1, 6)
        self.task_name_tensor = torch.ones(1, 1, 6)
        self.softmax = torch.nn.functional.softmax

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = torch.nn.Linear
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("_ih") or pn.endswith("_hh"):
                    # all recurrent weights will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("_emb") or "_token" in pn:
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def forward(self, idx, target=None, target_point=None, light_hazard=None):

        if self.config_all.model.pre_training.get("pretraining", "none") == "none":
            assert (
                    target_point is not None
            ), "target_point must be provided for wp output"
            assert (
                    light_hazard is not None
            ), "light_hazard must be provided for wp output"

        # create batch of same size as input
        x_batched = torch.cat(idx, dim=0)
        input_batch = self.pad_sequence_batch(x_batched)
        input_batch_type = input_batch[:, :, 0]  # car or map
        input_batch_data = input_batch[:, :, 1:]

        if self.add_learnable_input:
            input_batch_data = torch.cat((input_batch_data, self.task_name_tensor.cuda(input_batch_data.device).
                                          repeat(input_batch_data.size(0), 1, 1)), 1)

            input_batch_type = torch.cat((input_batch_type, (15 * torch.ones(input_batch_type.size(0), 1).
                                                             cuda(input_batch_type.device))), 1)

        # create same for output in case of multitask training to use this as ground truth
        if target is not None:
            y_batched = torch.cat(target, dim=0)
            output_batch = self.pad_sequence_batch(y_batched)
            output_batch_type = output_batch[:, :, 0]  # car or map
            output_batch_data = output_batch[:, :, 1:]

        # create masks by object type
        car_mask = (input_batch_type == 1).unsqueeze(-1)
        route_mask = (input_batch_type == 2).unsqueeze(-1)
        other_mask = torch.logical_and(route_mask.logical_not(), car_mask.logical_not())
        masks = [car_mask, route_mask, other_mask]

        # get size of input
        (B, O, A) = (input_batch_data.shape)  # batch size, number of objects, number of attributes

        # embed tokens object wise (one object -> one token embedding)
        input_batch_data = rearrange(
            input_batch_data, "b objects attributes -> (b objects) attributes"
        )
        embedding = self.tok_emb(input_batch_data)
        embedding = rearrange(embedding, "(b o) features -> b o features", b=B, o=O)

        # create object type embedding
        obj_embeddings = [
            self.obj_emb[i](self.obj_token[i]) for i in range(self.object_types)
        ]  # list of a tensors of size 1 x features

        # add object type embedding to embedding (mask needed to only add to the correct tokens)
        embedding = [
            (embedding + obj_embeddings[i]) * masks[i] for i in range(self.object_types)
        ]
        embedding = torch.sum(torch.stack(embedding, dim=1), dim=1)

        # embedding dropout
        x = self.drop(embedding)

        # Transformer Encoder; use embedding for hugging face model and get output states and attention map
        output = self.model(**{"inputs_embeds": embedding}, output_attentions=True)

        if self.add_learnable_input:
            task_name_hidden_state = output.last_hidden_state[:, -1]
            pred_task_name = self.softmax(self.task_name_layer(task_name_hidden_state), dim=1)
            output.last_hidden_state = output.last_hidden_state[:, :-1]

        x, attn_map = output.last_hidden_state, output.attentions

        # forecasting encoding
        if (
                self.config_all.model.pre_training.get("pretraining", "none") == "forecast"
                or self.config_all.model.training.get("pretraining_path", "none") != "none"
        ):
            car_mask_output = (output_batch_type == 1).unsqueeze(-1)
            non_car_mask_output = (output_batch_type != 1).unsqueeze(-1)
            # size: list of self.num_attributes tensors of size B x O x vocab_size (vocab_size differs for each attribute)
            # we do forecasting only for vehicles
            logits = [
                self.heads[i](x) * car_mask_output - 999 * non_car_mask_output
                for i in range(self.num_attributes)
            ]

            logits_list = []
            for i in range(self.num_attributes):
                element = self.heads[i](x) * car_mask_output - 999 * non_car_mask_output
                logits_list.append(element)

                asd = 0

            #### wp of other actors
            """input_tensor_batch_list = []
            tensor_car_mask_output = car_mask_output.squeeze(-1)
            for batch_index, tensor in enumerate(input_batch[:, :, 1:]):
                input_tensor_batch_list.append(tensor[tensor_car_mask_output[batch_index]][:, :2])
            input_tensor_batch = torch.stack(input_tensor_batch_list, 1)"""

            try:
                input_cars_info = input_batch_data[(input_batch_type.squeeze(0) == 1)]
            except:
                input_cars_info = []

            batch_size = x[:, 1:, :].size(0)
            other_z = self.other_wp_head(x)
            hidden_other_z = other_z
            other_output = []
            other_x = input_batch[:, :, 1:][:, :,
                      :2]  # torch.zeros(size=(hidden_other_z.shape[0], hidden_other_z.shape[1], 2),
            # dtype=other_z.dtype).type_as(hidden_other_z)
            other_x = other_x.reshape(-1, 2)
            hidden_other_z = hidden_other_z.reshape(-1, 512)
            for _ in range(4):  ### other actor wp sequence len
                hidden_other_z = self.other_wp_decoder(other_x, hidden_other_z)
                d_other_x = self.other_wp_relu(hidden_other_z)
                d_other_x = self.other_wp_output(d_other_x)
                other_x = d_other_x + other_x
                other_output.append(other_x.reshape(batch_size, -1, 2))

            asd = 0

            other_output_list = []
            tensor_car_mask_output = car_mask_output.squeeze(-1)
            for _, tensor_batch in enumerate(other_output):
                tensor_batch_list = []
                for tensor_index, tensor in enumerate(tensor_batch):
                    tensor_batch_list.append(tensor[tensor_car_mask_output[tensor_index]])
                paded_tensor = torch.swapaxes(pad_sequence(tensor_batch_list), 0, 1)
                other_output_list.append(paded_tensor)
            # paded_other_output_tensor = torch.swapaxes(pad_sequence(other_output_list), 0, 1)

            new_other_output = torch.stack(other_output_list, 1)

            assert (new_other_output[0][0] == other_output_list[0][0]).all()

            self.dummy_wp_tensor = torch.zeros(new_other_output.size(0), 4, 40, 2)

            other_gru_wp_head = torch.cat((new_other_output,
                                           self.dummy_wp_tensor[:, :,
                                           0:self.dummy_wp_tensor.size(2) - new_other_output.size(2), :].cuda(
                                               new_other_output.device)), 2)

            other_gru_wp_head = other_gru_wp_head.view(new_other_output.size(0), -1).detach()

            other_gru_wp_head = self.other_gru_wp_head_layer(other_gru_wp_head)

            logits = [
                rearrange(logit, "b o vocab_size -> (b o) vocab_size")
                for logit in logits
            ]

            # get target (GT) in same shape as logits
            targets = [
                output_batch_data[:, :, i].unsqueeze(-1) * car_mask_output - 999 * non_car_mask_output
                for i in range(self.num_attributes)
            ]
            asd = 0

            targets = [
                rearrange(target, "b o vocab_size -> (b o) vocab_size").long()
                for target in targets
            ]
            asd = 0

            # if we do pre-training (not multitask) we don't need wp for pre-trining step so we can return here
            if (
                    self.config_all.model.pre_training.get("pretraining", "none") == "forecast"
                    and self.config_all.model.pre_training.get("multitask", False) == False
            ):
                return logits, targets
        else:
            logits = None

        # get waypoint predictions
        z = self.wp_head(x[:, 0, :])
        # add traffic ligth flag
        z = torch.cat((z, light_hazard, other_gru_wp_head), 1)

        output_wp, other_vehicle_image = self.main_gru(z, target_point, new_other_output.squeeze(0), input_cars_info,
                                                       input_batch_data, route_mask)

        pred_wp = torch.stack(output_wp, dim=1)

        # pred the wapoints in the vehicle coordinate and we convert it to lidar coordinate here because the GT waypoints is in lidar coordinate
        pred_wp[:, :, 0] = pred_wp[:, :, 0] - 1.3

        paded_other_output_list = []


        if (
                self.config_all.model.pre_training.get("pretraining", "none") == "forecast"
                and self.config_all.model.pre_training.get("multitask", False) == True
        ):
            return logits, targets, pred_wp, attn_map, new_other_output, output_batch_type, input_cars_info, pred_task_name, self.dummy_collision_label, other_vehicle_image
        else:
            return logits, targets, pred_wp, attn_map

    def draw_route(self, input_batch_data, route_mask):

        route_array = ((input_batch_data.squeeze(0)[route_mask.squeeze(0).squeeze(-1)][:, 0:2].to(torch.int32).cpu().numpy()[:, ::-1]* -4) + 100)
        route_mask_wp = np.zeros((200, 200)).astype(np.uint8)

        for index, wp in enumerate(route_array):
            if index+1 != len(route_array):
                cv2.line(route_mask_wp, tuple(route_array[index]), tuple(route_array[index+1]), (255), 3)

        cv2.imwrite('route_mask_wp.png', route_mask_wp)

        return route_mask_wp

    def route_measure(self, is_there_vehicle, current_wp_image_list, route_mask_wp):

        sum_route_intersection = 0
        for current_wp in current_wp_image_list:
            sum_route_intersection += int(np.sum(current_wp*route_mask_wp) > 0)

        return max(sum_route_intersection/len(current_wp_image_list), is_there_vehicle)

    def main_gru(self, z, target_point, new_other_output, input_cars_info, input_batch_data, route_mask, sigma=0.3):

        route_mask_wp = self.draw_route(input_batch_data, route_mask)

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype)
        x = x.type_as(z)

        output_wp = self.get_wp(x, z, target_point, new_other_output, input_cars_info, sigma)
        output_wp_list = self.tensor2numpy(output_wp)


        deterministic_dx, collision, current_wp_image, other_vehicle_image, current_wp_image_list = self.collision_test(new_other_output, input_cars_info, np.array(output_wp_list).squeeze(1))
        print("current_wp_image*other_vehicle_image: ", np.sum(current_wp_image*other_vehicle_image))
        cv2.imwrite('current_wp_image.png', current_wp_image)
        cv2.imwrite('other_vehicle_image.png', other_vehicle_image)



        is_there_vehicle = 1 - int(np.sum(other_vehicle_image*route_mask_wp) > 0)

        route_measure_value = self.route_measure(is_there_vehicle, current_wp_image_list, route_mask_wp) #between 0-1



        self.dummy_collision_label = collision

        if True:
            output_wp_candidate_list = []
            magnitudes_list = []

            if not collision:
                distance = np.mean(output_wp_list, 0) - target_point.cpu().numpy().squeeze(0)
                magnitudes_list.append(np.linalg.norm(distance, axis=1) + (1-route_measure_value))
                output_wp_candidate_list.append(output_wp)

            for _ in range(50):
                output_wp_candidate = self.get_wp(x, z, target_point, sigma, new_other_output, input_cars_info,
                                                  deterministic=False)
                np_output_wp_candidate = self.tensor2numpy(output_wp_candidate)
                _, candidate_collision,_, _, current_wp_image_list = \
                    self.collision_test(new_other_output, input_cars_info, np.array(np_output_wp_candidate).squeeze(1))

                route_measure_value = self.route_measure(is_there_vehicle, current_wp_image_list,
                                                         route_mask_wp)  # between 0-1
                if not candidate_collision:
                    distance = np.mean(np_output_wp_candidate, 0) - target_point.cpu().numpy().squeeze(0)
                    magnitudes_list.append(np.linalg.norm(distance, axis=1) + (1-route_measure_value))
                    output_wp_candidate_list.append(output_wp_candidate)

            if len(output_wp_candidate_list) != 0:
                output_wp = output_wp_candidate_list[np.argmin(magnitudes_list)]#output_wp_candidate
            else:
                stop_output_wp = []
                for _ in range(4):
                    stop_output_wp.append(torch.tensor([[0, 0]]).cuda(output_wp[0].device).to(output_wp[0].dtype))
                output_wp = stop_output_wp


        other_vehicle_image = cv2.cvtColor(other_vehicle_image, cv2.COLOR_GRAY2BGR)
        other_vehicle_image[current_wp_image.astype(np.bool)] = (255, 0, 0)

        return output_wp, other_vehicle_image

    def tensor2numpy(self, output_wp):
        output_wp_list = []
        for wp in output_wp:
            output_wp_list.append(wp.cpu().numpy())

        return output_wp_list


    def get_wp(self, x, z, target_point, sigma, new_other_output, input_cars_info, deterministic=True):
        output_wp = list()

        for _ in range(self.config_all.model.training.pred_len):
            x_in = torch.cat([x, target_point], dim=1)
            z = self.wp_decoder(x_in, z)
            dx = self.wp_output(z)

            if not deterministic:
                dx_list = self.sample_dx(dx + x, sigma)
                #dx_list, collision = self.collision_test(new_other_output, input_cars_info,
                #                                         dx_list)

                distance = np.array(dx_list) - target_point.cpu().numpy().squeeze(0)
                magnitudes = np.linalg.norm(distance, axis=1)
                arg_index = np.argmin(magnitudes)
                dx = dx_list[arg_index]


            x = torch.tensor(dx).cuda(x.device) + x
            output_wp.append(x)

        return output_wp

    def sample_dx(self, dx, sigma, sample_number=50):
        dx_list = [] #[dx[0][0].cpu(), dx[0][1].cpu()]
        for _ in range(sample_number):
            sampled_x = np.random.normal(loc=dx[0][0].cpu(), scale=sigma)
            sampled_y = np.random.normal(loc=dx[0][1].cpu(), scale=sigma)
            new_dx = [sampled_x, sampled_y]
            dx_list.append(new_dx)

        return dx_list

    def collision_test(self, points, input_cars_info, sample_points,  speed_max=5, width=6.5, height=16.5, thickness=2):
        collision = False

        other_vehicle_image = np.zeros((200, 200, 3), dtype=np.uint8)#.cpu().numpy().astype(np.uint32)
        other_vehicle_above_image = np.zeros((200, 200), dtype=np.uint8)

        if points.size(1) != 0:
            new_other_output_list = []
            for gru_index in range(points.size(1)):
                new_other_output_list.append(points[:, gru_index].cpu())

            for gru_index, vehicle in enumerate(new_other_output_list):
                other_vehicle_image, other_vehicle_above_image = self.draw_pred_wp_gru(other_vehicle_image,
                                  copy.deepcopy(vehicle), input_cars_info[gru_index])

        other_vehicle_image = cv2.cvtColor(other_vehicle_image, cv2.COLOR_BGR2GRAY)
        non_collided_wp_list = []
        center_list = []
        current_wp_image = np.zeros((200, 200), dtype=np.uint8)
        current_wp_image_list = []

        for index, point in enumerate(sample_points):
            center = point
            center[0] = center[0] * (-1)
            center = center * 4 + 100
            center_list.append(center[::-1])

            #for route measure mask
            if index >= 1:
                current_indv_wp_image = np.zeros((200, 200), dtype=np.uint8)
                cv2.circle(current_indv_wp_image, tuple(center[::-1]), 1, (255), -1)
                current_wp_image_list.append(current_indv_wp_image)

        center_list = np.array(center_list)
        start_point = tuple(center_list[0])
        end_point = tuple(center_list[-1])  # Loop back to the first point
        cv2.line(current_wp_image, start_point, end_point, (255, 0, 0), 3)

        """if np.sum(current_wp_image * other_vehicle_above_image) >= 0:
            intersection = int(((1 - (min(current_speed, speed_max) / speed_max)) *
                            (np.sum(current_wp_image * other_vehicle_above_image))) > 0.5)

        elif np.sum(current_wp_image*other_vehicle_image) >= 0:
            intersection = 1
        else:
            intersection = 0"""

        """if intersection == 1:
            collision = True"""


        if np.sum(current_wp_image*other_vehicle_image) == 0:
            non_collided_wp_list.append(sample_points)
        else:
            collision = True






        return sample_points, collision, current_wp_image, other_vehicle_image, current_wp_image_list



    def pad_sequence_batch(self, x_batched):
        """
        Pads a batch of sequences to the longest sequence in the batch.
        """
        # split input into components
        x_batch_ids = x_batched[:, 0]

        x_tokens = x_batched[:, 1:]

        B = int(x_batch_ids[-1].item()) + 1
        input_batch = []
        for batch_id in range(B):
            # get the batch of elements
            x_batch_id_mask = x_batch_ids == batch_id

            # get the batch of types
            x_tokens_batch = x_tokens[x_batch_id_mask]

            x_seq = torch.cat([self.cls_emb, x_tokens_batch, self.eos_emb], dim=0)

            input_batch.append(x_seq)

        padded = torch.swapaxes(pad_sequence(input_batch), 0, 1)
        input_batch = padded[:B]

        return input_batch

    def control_pid(self, waypoints, velocity, is_stuck=False):
        """Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        """
        assert waypoints.size(0) == 1
        waypoints = waypoints[0].data.cpu().numpy()
        # when training we transform the waypoints to lidar coordinate, so we need to change is back when control
        waypoints[:, 0] += 1.3


        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        if is_stuck:
            desired_speed = np.array(4.0)  # default speed of 14.4 km/h

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0
        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90
        if brake:
            angle = 0.0
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        return steer, throttle, brake

    def get_new_center(self, index, input_cars_info, center, offset, scale_1, scale_2):

        center = (center - offset) / scale_1
        center = center * (-1)
        center = center * scale_2 + offset  # + np.array([100, 100])

        if type(input_cars_info) != type(None):
            ps_0 = np.array([input_cars_info[0].cpu(), input_cars_info[1].cpu()])
            ps_0[0] = ps_0[0] * (-1)
            ps_0[1] = ps_0[1] * (-1)
            ps_0 = ps_0 * 4 + np.array([100, 100])
            ps_0 = torch.tensor(ps_0)

            if index == 0:
                self.diff_offset = center - ps_0
                new_center = ps_0
            else:
                new_center = center - self.diff_offset
        else:
            new_center = center

        return new_center

    def draw_pred_wp_gru(self, plant_input_image, pred_wp, input_cars_info=None, width=4.5, height=14.5,
                             thickness=2, scale_1=40, scale_2=90,
                             offset=torch.tensor([73, 73]), offset_2=torch.tensor([2, 2])):

        mask_wp = np.zeros((200, 200)).astype(np.uint8)
        other_vehicle_above = np.zeros((200, 200)).astype(np.uint8)

        for index, center in enumerate(pred_wp):

            new_center = self.get_new_center(index, input_cars_info, center, offset, scale_1, scale_2)

            top_left = (int(new_center[1] - width / 2), int(new_center[0] - height / 2))
            bottom_right = (int(new_center[1] + width / 2), int(new_center[0] + height / 2))

            # Draw the rectangle (bounding box)
            if index < 3:
                line_mask_wp = np.zeros((200, 200)).astype(np.uint8)
                new_center_t1 = self.get_new_center(index, input_cars_info, pred_wp[index+1], offset, scale_1, scale_2)
                cv2.line(line_mask_wp, tuple(new_center.to(torch.int32)), tuple(new_center_t1.to(torch.int32)), (255, 0, 0), 3)
                line_mask_wp = cv2.rotate(line_mask_wp, cv2.ROTATE_90_CLOCKWISE)
                line_mask_wp = cv2.flip(line_mask_wp, 1)
                mask_wp += line_mask_wp

            cv2.rectangle(mask_wp, bottom_right, top_left, (255), int(thickness))

            if center[0] > 100:
                cv2.rectangle(other_vehicle_above, bottom_right, top_left, (255), int(thickness))

            plant_input_image[mask_wp.astype(np.bool)] = (255, 255, 255)

        return plant_input_image, other_vehicle_above


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative
