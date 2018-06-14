# -*- coding: utf-8 -*-
"""
    File name: config.py
    Project:
    Desciption:
    Author: Sijia Liu (m142167)
    Date created: Feb 13, 2018
"""

from ConfigParser import SafeConfigParser


class ChemProtConfig:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = SafeConfigParser()
        self.config.read(self.config_path)

    def get_stage_config(self):
        return self.config

    def get_target_labels(self):
        """
        skip no relations (label 0)
        :param config_path:
        :return:
        """
        rel2id = self.read_rel_id()
        target_labels = ['NA'] * len(rel2id.keys())
        for label, index in rel2id.iteritems():
            target_labels[int(index)] = label

        return target_labels

    def read_rel_id(self):
        relation2id = {}
        for k, v in self.config.items('rel_label'):
            if k == 'na':
                relation2id['NA'] = int(v)
            else:
                # hard code to avoid ":" issue of SafeConfigParser,
                # which auto use ":" to separate config keys and values.
                relation2id['CPR:' + k[-1]] = int(v)

        return relation2id

    def get(self, stage, key):
        return self.config.get(stage, key)
