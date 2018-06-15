import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

def SAN(input_list, ad_net_list, grl_layer_list, class_weight, use_gpu=True):
    loss = 0
    outer_product_out = torch.bmm(input_list[0].unsqueeze(2), input_list[1].unsqueeze(1))
    batch_size = input_list[0].size(0) // 2
    dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
    if use_gpu:
        dc_target = dc_target.cuda()
    for i in range(len(ad_net_list)):
        ad_out = ad_net_list[i](grl_layer_list[i](outer_product_out.narrow(2, i, 1).squeeze(2)))
        loss += nn.BCELoss()(ad_out.view(-1), dc_target.view(-1))
    return loss
