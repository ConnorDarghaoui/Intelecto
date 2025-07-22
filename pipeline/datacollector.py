# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from collections import defaultdict


class Result(object):
    def __init__(self):
        self.res_dict = {}

    def update(self, res, name):
        self.res_dict[name] = res

    def get(self, name):
        if name in self.res_dict:
            return self.res_dict[name]
        return None

    def clear(self, name):
        if name in self.res_dict:
            del self.res_dict[name]


class DataCollector(object):
    """
    DataCollector of pphuman pipeline, it will collect and handle the data
    """

    def __init__(self):
        #store all data by frame_id as key
        self.mots = {}  #storage mot results
        self.mot_keypoint17 = {}  #storage mot keypoint results
        self.mot_action = {}  #storage mot action results
        self.det_action = {}  #storage det action results
        self.cls_action = {}  #storage cls action results
        self.vehicleplate = {}  #storage vehicleplate results
        self.vehicle_attr = {}  #storage vehicle_attr results
        self.human_attr = {}  #storage human_attr results
        self.vehicle_retrograde = {}  #storage vehicle_retrograde results

    def append(self, frame_id, result):
        """
        add correspond results to different categories by frame_id as key
        """
        mot_res = result.get('mot')
        if mot_res is not None:
            self.mots[frame_id] = mot_res

        kpt_res = result.get('kpt')
        if kpt_res is not None:
            self.mot_keypoint17[frame_id] = kpt_res

        mot_action_res = result.get('skeleton_action')
        if mot_action_res is not None:
            self.mot_action[frame_id] = mot_action_res

        det_action_res = result.get('det_action')
        if det_action_res is not None:
            self.det_action[frame_id] = det_action_res

        cls_action_res = result.get('cls_action')
        if cls_action_res is not None:
            self.cls_action[frame_id] = cls_action_res

        vehicleplate_res = result.get('vehicleplate')
        if vehicleplate_res is not None:
            self.vehicleplate[frame_id] = vehicleplate_res

        vehicle_attr_res = result.get('vehicle_attr')
        if vehicle_attr_res is not None:
            self.vehicle_attr[frame_id] = vehicle_attr_res

        human_attr_res = result.get('attr')
        if human_attr_res is not None:
            self.human_attr[frame_id] = human_attr_res

        vehicle_retrograde_res = result.get('vehicle_retrograde')
        if vehicle_retrograde_res is not None:
            self.vehicle_retrograde[frame_id] = vehicle_retrograde_res

    def get_res(self):
        """
        return all results
        """
        return {
            'mots': self.mots,
            'kpt': self.mot_keypoint17,
            'mot_action': self.mot_action,
            'det_action': self.det_action,
            'cls_action': self.cls_action,
            'vehicleplate': self.vehicleplate,
            'vehicle_attr': self.vehicle_attr,
            'human_attr': self.human_attr,
            'vehicle_retrograde': self.vehicle_retrograde
        }

    def get_carlp(self, trackid):
        """
        get carlicense plate by track_id
        """
        for frame_id in self.vehicleplate.keys():
            vehicleplate_info = self.vehicleplate[frame_id]
            plate = vehicleplate_info['vehicleplate']
            for plate_res, track_id_res in zip(plate,
                                               vehicleplate_info['track_id']):
                if track_id_res == trackid:
                    return plate_res
        return None
