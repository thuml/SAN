import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
import torch.utils.data as util_data
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable

optim_dict = {"SGD": optim.SGD}

def image_classification_predict(loader, model, test_10crop=True, gpu=True, softmax_param=1.0):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['target'+str(i)]) for i in range(10)]
        for i in range(len(loader['target0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = Variable(inputs[j].cuda())
                labels = Variable(labels.cuda())
            else:
                for j in range(10):
                    inputs[j] = Variable(inputs[j])
                labels = Variable(labels)
            outputs = []
            for j in range(9):
                _, predict_out = model(inputs[j])
                outputs.append(nn.Softmax(dim=1)(softmax_param * predict_out))
            _, outputs_center = model(inputs[9])
            outputs.append(nn.Softmax(dim=1)(softmax_param * outputs_center))
            softmax_outputs = sum(outputs)
            outputs = outputs_center
            if start_test:
                all_output = outputs.data.float()
                all_softmax_output = softmax_outputs.data.cpu().float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        iter_val = iter(loader["target_test"])
        for i in range(len(loader['target_test'])):
            data = iter_val.next()
            inputs = data[0]
            if gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            _, outputs = model(inputs)
            softmax_outputs = nn.Softmax(dim=1)(softmax_param * outputs)
            if start_test:
                all_output = outputs.data.cpu().float()
                all_softmax_output = softmax_outputs.data.cpu().float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    _, predict = torch.max(all_output, 1)
    return all_softmax_output, predict, all_output, all_label

def image_classification_test(loader, model, test_10crop=True, gpu=True):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test'+str(i)]) for i in range(10)]
        for i in range(len(loader['test0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = Variable(inputs[j].cuda())
                labels = Variable(labels.cuda())
            else:
                for j in range(10):
                    inputs[j] = Variable(inputs[j])
                labels = Variable(labels)
            outputs = []
            for j in range(10):
                _, predict_out = model(inputs[j])
                outputs.append(nn.Softmax(dim=1)(predict_out))
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)       
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label) / float(all_label.size()[0])
    return accuracy


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
    prep_dict["target"] = prep.image_train( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
    else:
        prep_dict["test"] = prep.image_test( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
               
    ## set loss
    class_criterion = nn.CrossEntropyLoss()
    transfer_criterion = loss.SAN
    loss_params = config["loss"]

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = util_data.DataLoader(dsets["source"], \
            batch_size=data_config["source"]["batch_size"], \
            shuffle=True, num_workers=4)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = util_data.DataLoader(dsets["target"], \
            batch_size=data_config["target"]["batch_size"], \
            shuffle=True, num_workers=4)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"+str(i)] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"]["val"+str(i)])
            dset_loaders["test"+str(i)] = util_data.DataLoader(dsets["test"+str(i)], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=4)
            dsets["target"+str(i)] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["test"]["val"+str(i)])
            dset_loaders["target"+str(i)] = util_data.DataLoader(dsets["target"+str(i)], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=4)
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=4)
        dsets["target_test"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["target_test"] = util_data.DataLoader(dsets["target_test"], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=4)

    class_num = config["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()

    ## collect parameters
    if net_config["params"]["new_cls"]:
        if net_config["params"]["use_bottleneck"]:
            parameter_list = [{"params":base_network.feature_layers.parameters(), "lr":1}, \
                            {"params":base_network.bottleneck.parameters(), "lr":10}, \
                            {"params":base_network.fc.parameters(), "lr":10}]
        else:
            parameter_list = [{"params":base_network.feature_layers.parameters(), "lr":1}, \
                            {"params":base_network.fc.parameters(), "lr":10}]
    else:
        parameter_list = [{"params":base_network.parameters(), "lr":1}]

    ## add additional network for some methods
    ad_net_list = []
    gradient_reverse_layer_list = []
    class_weight = torch.from_numpy(np.array([1.0] * class_num))
    if use_gpu:
        class_weight = class_weight.cuda()
    for i in range(class_num):
        if config["dataset"] == "caltech":
            ad_net = network.SmallAdversarialNetwork(base_network.output_num())
        elif config["dataset"] == "imagenet":
            ad_net = network.LittleAdversarialNetwork(base_network.output_num())
        else:
            ad_net = network.AdversarialNetwork(base_network.output_num())
        gradient_reverse_layer = network.AdversarialLayer()
        gradient_reverse_layer_list.append(gradient_reverse_layer)
        if use_gpu:
            ad_net = ad_net.cuda()
        ad_net_list.append(ad_net)
        parameter_list.append({"params":ad_net.parameters(), "lr":10})
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]


    ## train   
    len_train_source = len(dset_loaders["source"]) - 1
    len_train_target = len(dset_loaders["target"]) - 1
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == 0:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, \
                    base_network, test_10crop=prep_config["test_10crop"], \
                    gpu=use_gpu)
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str)
            config["out_file"].flush()
            print(log_str)
        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                "iter_{:05d}_model.pth.tar".format(i)))
        
        if (i % loss_params["update_iter"] == loss_params["update_iter"] - 1):
            base_network.train(False)
            target_fc8_out, _, target_before_softmax, target_label = image_classification_predict(dset_loaders, base_network, softmax_param=1.0)
            class_weight = torch.mean(target_fc8_out, 0)
            class_weight = (class_weight / torch.max(class_weight)).cuda().view(-1)
            for j in range(class_num):
                gradient_reverse_layer_list[j].high = class_weight[j]
        

        ## train one iter
        base_network.train(True)
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        if use_gpu:
            inputs_source, inputs_target, labels_source = \
                Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), \
                Variable(labels_source).cuda()
        else:
            inputs_source, inputs_target, labels_source = Variable(inputs_source), \
                Variable(inputs_target), Variable(labels_source)
           
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        features, outputs = base_network(inputs)

        softmax_out = nn.Softmax(dim=1)(outputs).detach()
        for ad_net in ad_net_list:
            ad_net.train(True)
        transfer_loss = transfer_criterion([features, softmax_out], ad_net_list, \
            gradient_reverse_layer_list, class_weight, use_gpu)
        entropy_loss = loss_params["entropy_trade_off"] * \
                        loss.EntropyLoss(nn.Softmax(dim=1)(outputs))

        classifier_loss = class_criterion(outputs.narrow(0, 0, inputs.size(0)/2), labels_source)

        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + entropy_loss
        total_loss.backward()
        optimizer.step()
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Partial Transfer Learning with Selective Adversarial Networks')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--dset', type=str, default='Office', help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../data/office/amazon_31_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../data/office/webcam_10_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 


    # train config
    config = {}
    config["num_iterations"] = 50004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_path"] = "../snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    if "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, "params":{"resnet_name":args.net, \
                            "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "AlexNet" in args.net:
        config["network"] = {"name":network.AlexNetFc, "params":{"use_bottleneck":True, \
                            "bottleneck_dim":256, "new_cls":True} }
    config["prep"] = {"test_10crop":True, "resize_size":256, "crop_size":224}
    config["loss"] = {"trade_off":1.0, "entropy_trade_off":0.2, "update_iter":200}
    config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"init_lr":0.0003, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    if config["dataset"] == "office":
        config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                          "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                          "test":{"list_path":args.t_dset_path, "batch_size":4}}
        config["class_num"] = 31
    elif config["dataset"] == "imagenet":
        config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                          "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                          "test":{"list_path":args.t_dset_path, "batch_size":4}}
        config["loss"] = {"trade_off":1.0, "entropy_trade_off":1.0, "update_iter":200}
        config["network"]["params"]["use_bottleneck"] = False
        config["network"]["params"]["new_cls"] = False
        config["class_num"] = 1000
    elif config["dataset"] == "caltech":
        config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                          "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                          "test":{"list_path":args.t_dset_path, "batch_size":4}}
        config["loss"] = {"trade_off":1.0, "entropy_trade_off":0.1, "update_iter":200}
        config["optimizer"]["lr_param"]["init_lr"] = 0.001
        config["class_num"] = 256
    train(config)
