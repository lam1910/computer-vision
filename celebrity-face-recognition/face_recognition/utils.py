import torch
from copy import deepcopy


def load_weight(weights_path, state_dict):
    state_dict = deepcopy(state_dict)
    pretrained_weight = torch.load(weights_path)
    state_dict['block1.0.weight'] = pretrained_weight['conv1_1.weight']
    state_dict['block1.0.bias'] = pretrained_weight['conv1_1.bias']
    state_dict['block1.2.weight'] = pretrained_weight['conv1_2.weight']
    state_dict['block1.2.bias'] = pretrained_weight['conv1_2.bias']
    state_dict['block2.0.weight'] = pretrained_weight['conv2_1.weight']
    state_dict['block2.0.bias'] = pretrained_weight['conv2_1.bias']
    state_dict['block2.2.weight'] = pretrained_weight['conv2_2.weight']
    state_dict['block2.2.bias'] = pretrained_weight['conv2_2.bias']
    state_dict['block3.0.weight'] = pretrained_weight['conv3_1.weight']
    state_dict['block3.0.bias'] = pretrained_weight['conv3_1.bias']
    state_dict['block3.2.weight'] = pretrained_weight['conv3_2.weight']
    state_dict['block3.2.bias'] = pretrained_weight['conv3_2.bias']
    state_dict['block3.4.weight'] = pretrained_weight['conv3_3.weight']
    state_dict['block3.4.bias'] = pretrained_weight['conv3_3.bias']
    state_dict['block4.0.weight'] = pretrained_weight['conv4_1.weight']
    state_dict['block4.0.bias'] = pretrained_weight['conv4_1.bias']
    state_dict['block4.2.weight'] = pretrained_weight['conv4_2.weight']
    state_dict['block4.2.bias'] = pretrained_weight['conv4_2.bias']
    state_dict['block4.4.weight'] = pretrained_weight['conv4_3.weight']
    state_dict['block4.4.bias'] = pretrained_weight['conv4_3.bias']
    state_dict['block5.0.weight'] = pretrained_weight['conv5_1.weight']
    state_dict['block5.0.bias'] = pretrained_weight['conv5_1.bias']
    state_dict['block5.2.weight'] = pretrained_weight['conv5_2.weight']
    state_dict['block5.2.bias'] = pretrained_weight['conv5_2.bias']
    state_dict['block5.4.weight'] = pretrained_weight['conv5_3.weight']
    state_dict['block5.4.bias'] = pretrained_weight['conv5_3.bias']

    state_dict['block6.0.weight'] = pretrained_weight['fc6.weight']
    state_dict['block6.0.bias'] = pretrained_weight['fc6.bias']
    state_dict['block7.0.weight'] = pretrained_weight['fc7.weight']
    state_dict['block7.0.bias'] = pretrained_weight['fc7.bias']

    return state_dict
