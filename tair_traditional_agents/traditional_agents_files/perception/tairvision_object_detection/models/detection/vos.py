from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn.functional as F

from torch import nn, Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from tairvision_object_detection.models.detection.faster_rcnn import TwoMLPHead
from tairvision_object_detection.models.detection.mask_rcnn import MaskRCNNHeads

from tairvision_object_detection.ops import MultiScaleRoIAlign, sigmoid_focal_loss
from tairvision_object_detection.utils import reduce_mean


class VOSHead(nn.Module):

    def __init__(self, in_channels, num_classes,
                 roi_output_size=14, roi_sampling_ratio=2,
                 representation_size=1024
                 ):
        super(VOSHead, self).__init__()

        self.roi_align = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                            output_size=roi_output_size,
                                            sampling_ratio=roi_sampling_ratio)
        mask_layers = [256, 256, 256, 256]

        self.head = nn.Sequential(MaskRCNNHeads(in_channels, mask_layers, 1),
                                  TwoMLPHead(mask_layers[-1]*roi_output_size**2, representation_size)
                                  )

        #self.penultimate = nn.Sequential(nn.Linear(representation_size, representation_size), nn.ReLU())
        self.predictor = nn.Linear(representation_size, num_classes)
        self.energy_score_weights = nn.Parameter(torch.rand(num_classes, 1))

        self.id_class_queue_size = 1000
        self.id_data = nn.Parameter(torch.zeros(num_classes, self.id_class_queue_size, representation_size),
                                    requires_grad=False)
        self.id_data_counter = torch.zeros(num_classes, dtype=torch.int64)
        self.id_dist_dict = {}

        self.enable_vos = False
        self.vos_sample_size = 10000
        self.vos_select_size = 1

        self.dist_classifier = nn.Sequential(nn.Linear(1,512), nn.ReLU(),nn.Linear(512, 1))
        self.dist_loss_func = F.binary_cross_entropy_with_logits
        self.dist_loss_weight = 0.1

        self.num_classes = num_classes
        self.representation_size = representation_size

    def forward(self,
                features,           # type: List[Tensor]
                detections,         # type: List[Dict[str, Tensor]]
                image_shapes,       # type: List[Tuple[int, int]]
                targets=None,       # type: Optional[List[Dict[str, Tensor]]]
                ):

        boxes = [detection['boxes'] for detection in detections]

        box_features = self.roi_align(features, boxes, image_shapes)
        box_features = self.head(box_features)
        #penultimate_features = self.penultimate(box_features)
        scores = self.predictor(box_features)

        if targets is not None:
            id_labels = self.get_id_labels(targets).squeeze(1).cpu().numpy()

            # need to train a separate classifier to use later to get scores for vos_samples
            cls_loss = self.get_cls_loss(scores, id_labels)

            # maintaining queue for each class seperately
            id_data_updated = torch.zeros_like(self.id_data_counter)
            for i, id_label in enumerate(id_labels):
                id_data_updated[id_label] = 1
                if self.id_data_counter[id_label] < self.id_class_queue_size:
                    self.id_data[id_label, self.id_data_counter[id_label]] = box_features[i].detach()
                    self.id_data_counter[id_label] += 1
                else:
                    self.id_data[id_label] = torch.cat((self.id_data[id_label, 1:],
                                                        box_features[i:i + 1].detach()), dim=0)
            valid_ind = self.id_data_counter == 1000
            enable_vos = self.enable_vos and (valid_ind.sum().item() >= self.num_classes*0.70)

            # get class statistics
            mu, sigma = self.get_class_gaussians(self.id_data, valid_ind)

            # sample outliers from feature space
            if enable_vos:
                vos_samples = []
                valid_ind = valid_ind.nonzero()[:, 0].numpy()
                for i in valid_ind:
                    vos_sample = self.sample_vos(mu[i], sigma, id_data_updated[i], i)
                    vos_samples.append(vos_sample)
                vos_samples = torch.cat(vos_samples, dim=0)
            else:
                vos_samples = torch.zeros(self.num_classes, self.representation_size)

            # obtain dist_loss using id and vos samples
            scores_vos = self.predictor(vos_samples.to(scores.device))
            dist_loss = self.get_dist_loss(scores, scores_vos)

            dist_loss_weight = self.dist_loss_weight if enable_vos else 0
            dist_loss *= dist_loss_weight

            return cls_loss, dist_loss
        else:
            energy_scores = self.get_energy_score(scores, False)
            dist_logits = self.dist_classifier(energy_scores)

            boxes_per_image = [box.shape[0] for box in boxes]

            aux_scores, aux_labels = scores.sigmoid().max(dim=1)
            dist_scores, dist_labels = dist_logits.sigmoid().max(dim=1)

            aux_scores = aux_scores.split(boxes_per_image, dim=0)
            aux_labels = aux_labels.split(boxes_per_image, dim=0)
            dist_scores = dist_scores.split(boxes_per_image, dim=0)
            dist_labels = dist_labels.split(boxes_per_image, dim=0)

            for detection, aux_score, aux_label, dist_score, dist_label in \
                    zip(detections, aux_scores, aux_labels, dist_scores, dist_labels):
                detection['aux_scores'] = aux_score
                detection['aux_labels'] = aux_label
                detection['dist_scores'] = dist_score
                detection['dist_labels'] = dist_label
        return detections

    def get_id_labels(self, targets):

        labels = targets['labels']
        pos_inds_1d = torch.nonzero(labels.flatten() != self.num_classes).squeeze(1)
        id_labels = labels.flatten().view(-1, 1)[pos_inds_1d, :]

        return id_labels

    def get_class_gaussians(self, x, valid_ind):
        mu = x.mean(1)
        x_tilda = x - mu.unsqueeze(1)
        x_tilda = x_tilda[valid_ind]
        x_tilda = x_tilda.view(-1, x_tilda.shape[-1])

        # add the variance.
        sigma = torch.mm(x_tilda.t(), x_tilda) / len(x_tilda)
        # for stable training.
        sigma += 0.0001 * torch.eye(x_tilda.shape[-1]).to(x.device)

        return mu, sigma

    def sample_vos(self, mu, sigma, id_data_updated, dict_key):
        if id_data_updated or dict_key not in self.id_dist_dict.keys():
            new_dis = MultivariateNormal(mu, covariance_matrix=sigma)
            self.id_dist_dict[dict_key] = new_dis
        else:
            new_dis = self.id_dist_dict[dict_key]
        negative_samples = new_dis.rsample((self.vos_sample_size,))
        prob_density = new_dis.log_prob(negative_samples)

        # keep the data in the low density area.
        cur_samples, index_prob = torch.topk(-prob_density, self.vos_select_size)

        return negative_samples[index_prob]

    def get_cls_loss(self, scores_id, id_labels):
        # prepare one_hot
        cls_target = torch.zeros_like(scores_id)
        inds = range(len(scores_id))
        cls_target[inds, id_labels[inds]] = 1

        num_pos_local = torch.ones_like(scores_id[:, 0]).sum()
        num_pos_avg = max(reduce_mean(num_pos_local).item(), 1.0)

        cls_loss = sigmoid_focal_loss(scores_id, cls_target,
                                      alpha=0.25, gamma=2.0, reduction="sum"
                                      ) / num_pos_avg
        return cls_loss

    def get_dist_loss(self, scores, scores_vos):

        energy_scores = self.get_energy_score(scores)
        energy_scores_vos = self.get_energy_score(scores_vos)
        energy_scores_all = torch.cat([energy_scores, energy_scores_vos])

        dist_predictions = self.dist_classifier(energy_scores_all).squeeze(1)

        dist_labels = torch.cat([torch.ones_like(energy_scores),
                                 torch.zeros_like(energy_scores_vos)]).squeeze(1)
        dist_loss = self.dist_loss_func(dist_predictions, dist_labels)

        return dist_loss

    def get_energy_score(self, x, filter_empty_classes=True):

        valid_ind = self.id_data_counter > 0

        w_e = F.relu(self.energy_score_weights)
        if filter_empty_classes:
            x = x[:, valid_ind]
            w_e = w_e[valid_ind,]
        m, _ = torch.max(x, dim=1, keepdim=True)

        x_tilda = torch.exp(x - m)
        x_tilda = torch.mm(x_tilda, w_e)
        x_tilda = torch.log(x_tilda)

        return m + x_tilda


class VOSHeadAF(nn.Module):

    def __init__(self, num_classes,
                 score_func,
                 representation_size=256
                 ):
        super(VOSHeadAF, self).__init__()

        self.energy_score_weights = nn.Parameter(torch.rand(num_classes, 1))

        self.id_class_queue_size = 1000
        self.id_data = nn.Parameter(torch.zeros(num_classes, self.id_class_queue_size, representation_size),
                                    requires_grad=False)
        self.id_data_counter = torch.zeros(num_classes, dtype=torch.int64)
        self.id_dist_dict = {}

        self.enable_vos = False
        self.vos_sample_size = 10000
        self.vos_select_size = 1

        self.dist_classifier = nn.Sequential(nn.Linear(1,512), nn.ReLU(),nn.Linear(512, 1))
        self.dist_loss_func = F.binary_cross_entropy_with_logits
        self.dist_loss_weight = 0.1

        self.num_classes = num_classes
        self.representation_size = representation_size

        self.score_func = score_func

    def forward(self,
                detections,  # type: List[Dict[str, Tensor]]
                targets=None,  # type: Optional[List[Dict[str, Tensor]]]
                ):


        scores = [detection['cls_logits'] for detection in detections]
        detections_per_image = [score.shape[0] for score in scores]

        scores = torch.cat(scores, dim=0)

        if targets is not None:
            cls_feats = [detection['cls_feats'] for detection in detections]
            cls_feats = torch.cat(cls_feats, dim=0)

            id_labels = self.get_id_labels(targets).squeeze(1).cpu().numpy()

            # maintaining queue for each class seperately
            id_data_updated = torch.zeros_like(self.id_data_counter)
            for i, id_label in enumerate(id_labels):
                id_data_updated[id_label] = 1
                if self.id_data_counter[id_label] < self.id_class_queue_size:
                    self.id_data[id_label, self.id_data_counter[id_label]] = cls_feats[i].detach()
                    self.id_data_counter[id_label] += 1
                else:
                    self.id_data[id_label] = torch.cat((self.id_data[id_label, 1:],
                                                        cls_feats[i:i + 1].detach()), dim=0)
            valid_ind = self.id_data_counter == 1000
            enable_vos = self.enable_vos and (valid_ind.sum().item() >= self.num_classes*0.70)

            # get class statistics
            mu, sigma = self.get_class_gaussians(self.id_data, valid_ind)

            # sample outliers from feature space
            if enable_vos:
                vos_samples = []
                valid_ind = valid_ind.nonzero()[:, 0].numpy()
                for i in valid_ind:
                    vos_sample = self.sample_vos(mu[i], sigma, id_data_updated[i], i)
                    vos_samples.append(vos_sample)
                vos_samples = torch.cat(vos_samples, dim=0)
            else:
                vos_samples = torch.zeros(self.num_classes, self.representation_size)

            # obtain dist_loss using id and vos samples
            scores_vos = self.score_func(vos_samples.to(scores.device))
            dist_loss = self.get_dist_loss(scores, scores_vos)

            dist_loss_weight = self.dist_loss_weight if enable_vos else 0
            dist_loss *= dist_loss_weight

            return dist_loss
        else:
            energy_scores = self.get_energy_score(scores, False)
            dist_logits = self.dist_classifier(energy_scores)

            dist_scores = dist_logits.sigmoid()
            dist_scores = dist_scores.split(detections_per_image, dim=0)

            for detection, dist_score in zip(detections, dist_scores):
                detection['dist_scores'] = dist_score
        return detections

    def get_id_labels(self, targets):

        labels = targets['labels']
        pos_inds_1d = torch.nonzero(labels.flatten() != self.num_classes).squeeze(1)
        id_labels = labels.flatten().view(-1, 1)[pos_inds_1d, :]

        return id_labels

    def get_class_gaussians(self, x, valid_ind):
        mu = x.mean(1)
        x_tilda = x - mu.unsqueeze(1)
        x_tilda = x_tilda[valid_ind]
        x_tilda = x_tilda.view(-1, x_tilda.shape[-1])

        # add the variance.
        sigma = torch.mm(x_tilda.t(), x_tilda) / len(x_tilda)
        # for stable training.
        sigma += 0.0001 * torch.eye(x_tilda.shape[-1]).to(x.device)

        return mu, sigma

    def sample_vos(self, mu, sigma, id_data_updated, dict_key):
        if id_data_updated or dict_key not in self.id_dist_dict.keys():
            new_dis = MultivariateNormal(mu, covariance_matrix=sigma)
            self.id_dist_dict[dict_key] = new_dis
        else:
            new_dis = self.id_dist_dict[dict_key]
        negative_samples = new_dis.rsample((self.vos_sample_size,))
        prob_density = new_dis.log_prob(negative_samples)

        # keep the data in the low density area.
        cur_samples, index_prob = torch.topk(-prob_density, self.vos_select_size)

        return negative_samples[index_prob]

    def get_cls_loss(self, scores_id, id_labels):
        # prepare one_hot
        cls_target = torch.zeros_like(scores_id)
        inds = range(len(scores_id))
        cls_target[inds, id_labels[inds]] = 1

        num_pos_local = torch.ones_like(scores_id[:, 0]).sum()
        num_pos_avg = max(reduce_mean(num_pos_local).item(), 1.0)

        cls_loss = sigmoid_focal_loss(scores_id, cls_target,
                                      alpha=0.25, gamma=2.0, reduction="sum"
                                      ) / num_pos_avg
        return cls_loss

    def get_dist_loss(self, scores, scores_vos):

        energy_scores = self.get_energy_score(scores)
        energy_scores_vos = self.get_energy_score(scores_vos)
        energy_scores_all = torch.cat([energy_scores, energy_scores_vos])

        dist_predictions = self.dist_classifier(energy_scores_all).squeeze(1)

        dist_labels = torch.cat([torch.ones_like(energy_scores),
                                 torch.zeros_like(energy_scores_vos)]).squeeze(1)
        dist_loss = self.dist_loss_func(dist_predictions, dist_labels)

        return dist_loss

    def get_energy_score(self, x, filter_empty_classes=True):

        valid_ind = self.id_data_counter > 0

        w_e = F.relu(self.energy_score_weights)
        if filter_empty_classes:
            x = x[:, valid_ind]
            w_e = w_e[valid_ind,]
        m, _ = torch.max(x, dim=1, keepdim=True)

        x_tilda = torch.exp(x - m)
        x_tilda = torch.mm(x_tilda, w_e)
        x_tilda = torch.log(x_tilda)

        return m + x_tilda
