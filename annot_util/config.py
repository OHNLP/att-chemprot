# -*- coding: utf-8 -*-
"""
    File name: config.py
    Project: 
    Desciption: 
    Author: Sijia Liu (m142167)
    Date created: Feb 13, 2018
"""

from ConfigParser import SafeConfigParser


def get_stage_config(config_path):
    config = SafeConfigParser()
    config.read(config_path)

    return config


def get_target_labels(config_path):
    """
    skip no relations (label 0)
    :param config_path:
    :return:
    """
    rel2id = read_rel_id(config_path)
    target_labels = ['NA'] * len(rel2id.keys())
    for label, index in rel2id.iteritems():
        target_labels[int(index)] = label

    return target_labels


def read_rel_id(config_path):
    config = get_stage_config(config_path)
    relation2id = {}
    for k, v in config.items('rel_label'):
        if k == 'na':
            relation2id['NA'] = int(v)
        else:
            # hard code to avoid ":" issue of SafeConfigParser,
            # which auto use ":" to separate config keys and values.
            relation2id['CPR:' + k[-1]] = int(v)

    return relation2id


if __name__ == '__main__':
    # config = get_stage_config('config/main_config.ini')
    read_rel_id('config/main_config.ini')
    print get_target_labels('config/main_config.ini')