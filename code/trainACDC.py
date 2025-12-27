import argparse
import logging
import math
import os
import random
import shutil
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label
from utils.displacement import ABD_R_BCP3, process_image_batches, ABD_R_BCP
from dataloaders.dataset import (BaseDataSets, TwoStreamBatchSampler, WeakStrongAugment)
from networks.net_factory import BCP_net, net_factory
from utils import ramps, losses
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='D:\lyytest\ACDC\ABD-main\ABD-main\data\ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='BCP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--image_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=1, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int, default=6, help='multinum of random masks')
# patch size
parser.add_argument('--patch_size', type=int, default=64, help='patch_size4 16 64')
parser.add_argument('--h_size', type=int, default=4, help='h_size4')
parser.add_argument('--w_size', type=int, default=4, help='w_size4')
# top num
parser.add_argument('--top_num', type=int, default=4, help='top_num')
args = parser.parse_args()
dice_loss = losses.DiceLoss(n_classes=args.num_classes)

# def nt_xent_loss(z_i, z_j, temperature=0.5):
#     """ z_i, z_j: [B, D] 特征向量 """
#     B = z_i.size(0)
#     z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
#     sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # [2B, 2B]
#     sim_matrix = torch.exp(sim_matrix / temperature)
#     mask = ~torch.eye(2*B, dtype=torch.bool, device=z.device)
#     sim_matrix = sim_matrix[mask].view(2*B, -1)
#
#     pos_pairs = torch.exp(F.cosine_similarity(z_i, z_j, dim=1) / temperature)
#     pos_pairs = torch.cat([pos_pairs, pos_pairs], dim=0)  # 每个样本有两个正例
#
#     loss = -torch.log(pos_pairs / sim_matrix.sum(dim=1))
#     return loss.mean()

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()


def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)
    return probs


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x * 2 / (3 * shrink_param)), int(img_y * 2 / (3 * shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s * x_split, (x_s + 1) * x_split - patch_x)
            h = np.random.randint(y_s * y_split, (y_s + 1) * y_split - patch_y)
            mask[w:w + patch_x, h:h + patch_y] = 0
            loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y * 4 / 9)
    h = np.random.randint(0, img_y - patch_y)
    mask[h:h + patch_y, :] = 0
    loss_mask[:, h:h + patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)  # loss = loss_ce
    return loss_dice, loss_ce


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model = BCP_net(in_chns=1, class_num=num_classes)
    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([WeakStrongAugment(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, pin_memory=True)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            # -- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl,_ = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                  classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()



def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = max(0.0, min(1.0, current / rampup_length))
        return float(math.exp(-5.0 * (1.0 - current) ** 2))


def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model_1 = BCP_net(in_chns=1, class_num=num_classes)
    model_2 = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([WeakStrongAugment(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Train labeled {} samples".format(labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, pin_memory=True)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    load_net(ema_model, pre_trained_model)
    load_net_opt(model_1, optimizer1, pre_trained_model)
    load_net_opt(model_2, optimizer2, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model_1.train()
    model_2.train()
    ema_model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[
                                                                                               args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]

            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s, uimg_b_s = volume_batch_strong[
                                 args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch_strong[
                                                                                      args.labeled_bs + unlabeled_sub_bs:]
            lab_a_s, lab_b_s = label_batch_strong[:labeled_sub_bs], label_batch_strong[labeled_sub_bs:args.labeled_bs]

            with torch.no_grad():
                pre_a,feature_a = ema_model(uimg_a)
                pre_b,feature_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b = get_ACDC_masks(pre_b, nms=1)
                pre_a_s,_ = ema_model(uimg_a_s)
                pre_b_s,_ = ema_model(uimg_b_s)
                plab_a_s = get_ACDC_masks(pre_a_s, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b_s = get_ACDC_masks(pre_b_s, nms=1)
                img_mask, loss_mask = generate_mask(img_a)
                # unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                # l_label = lab_b * img_mask + ulab_b * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            net_input_unl_1 = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l_1 = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input_1 = torch.cat([net_input_unl_1, net_input_l_1], dim=0)

            net_input_unl_2 = uimg_a_s * img_mask + img_a_s * (1 - img_mask)
            net_input_l_2 = img_b_s * img_mask + uimg_b_s * (1 - img_mask)
            net_input_2 = torch.cat([net_input_unl_2, net_input_l_2], dim=0)

            _, feat_origa = model_1(volume_batch)
            _, feat_stronga = model_1(volume_batch_strong)
            # 在特征层面添加对比损失
            # 池化到向量
            feat_origa = F.adaptive_avg_pool2d(feat_origa, 1).view(feat_origa.size(0), -1)  # [B, D]
            feat_stronga = F.adaptive_avg_pool2d(feat_stronga, 1).view(feat_stronga.size(0), -1)  # [B, D]

            # 计算对比损失
            def feature_contrastive_loss(feat1, feat2, temperature=0.5):
                B = feat1.size(0)
                feat1 = F.normalize(feat1, dim=1)
                feat2 = F.normalize(feat2, dim=1)

                # 计算相似度矩阵
                similarity = F.cosine_similarity(feat1.unsqueeze(1), feat2.unsqueeze(0), dim=2) / temperature
                # similarity[i, j] = similarity between feat1[i] and feat2[j]

                # 标签：期望 feat1[i] 与 feat2[i] 最相似
                labels = torch.arange(B).to(feat1.device)  # [0, 1, 2, ..., B-1]

                # 交叉熵损失：最大化对角线元素
                loss = F.cross_entropy(similarity, labels)
                return loss
            # # 添加全局平均池化，将 [B, C, H, W] -> [B, C]
            # if len(feat_origa.shape) > 2:
            #     feat_origa = F.adaptive_avg_pool2d(feat_origa, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
            # if len(feat_stronga.shape) > 2:
            #     feat_stronga = F.adaptive_avg_pool2d(feat_stronga, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
            loss_contrastive1 = feature_contrastive_loss(feat_origa, feat_stronga)
            loss_contrastive2 = feature_contrastive_loss(feat_stronga, feat_origa)
            loss_contrastive = 0.5*(loss_contrastive1 +  loss_contrastive2 )
            # Model1 Loss
            out_unl_1,feature_unl_1 = model_1(net_input_unl_1)
            out_l_1, feature_l_1= model_1(net_input_l_1)
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            out_soft_1 = torch.softmax(out_1, dim=1)
            out_max_1 = torch.max(out_soft_1.detach(), dim=1)[0]
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False)
            unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            # Model2 Loss
            out_unl_2, feature_unl_2 = model_2(net_input_unl_2)
            out_l_2,feature_l_2 = model_2(net_input_l_2)
            out_2 = torch.cat([out_unl_2, out_l_2], dim=0)
            out_soft_2 = torch.softmax(out_2, dim=1)
            out_max_2 = torch.max(out_soft_2.detach(), dim=1)[0]
            out_pseudo_2 = torch.argmax(out_soft_2.detach(), dim=1, keepdim=False)
            unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_2, l_ce_2 = mix_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, u_weight=args.u_weight)
            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            # Model1 & Model2 Cross Pseudo Supervision
            pseudo_supervision1 = dice_loss(out_soft_1, out_pseudo_2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(out_soft_2, out_pseudo_1.unsqueeze(1))
            # ABD-R New Training Sample
            image_patch_last = ABD_R_BCP3(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args)
            image_output_1,_ = model_1(image_patch_last.unsqueeze(1))
            image_output_soft_1 = torch.softmax(image_output_1, dim=1)
            pseudo_image_output_1 = torch.argmax(image_output_soft_1.detach(), dim=1, keepdim=False)
            image_output_2,_ = model_2(image_patch_last.unsqueeze(1))
            image_output_soft_2 = torch.softmax(image_output_2, dim=1)
            pseudo_image_output_2 = torch.argmax(image_output_soft_2.detach(), dim=1, keepdim=False)
            # Model1 & Model2 Second Step Cross Pseudo Supervision
            pseudo_supervision3 = dice_loss(image_output_soft_1, pseudo_image_output_2.unsqueeze(1))
            pseudo_supervision4 = dice_loss(image_output_soft_2, pseudo_image_output_1.unsqueeze(1))

            # FFT
            image_patch_last_FFT = process_image_batches(net_input_1, net_input_2, 30)
            image_output_1_FFT,_ = model_1(image_patch_last_FFT.unsqueeze(1))
            image_output_soft_1_FFT = torch.softmax(image_output_1_FFT, dim=1)
            pseudo_image_output_1_FFT = torch.argmax(image_output_soft_1_FFT.detach(), dim=1, keepdim=False)
            image_output_2_FFT,_  = model_2(image_patch_last_FFT.unsqueeze(1))
            image_output_soft_2_FFT = torch.softmax(image_output_2_FFT, dim=1)
            pseudo_image_output_2_FFT = torch.argmax(image_output_soft_2_FFT.detach(), dim=1, keepdim=False)

            pseudo_supervision5 = dice_loss(image_output_soft_1_FFT, pseudo_image_output_2_FFT.unsqueeze(1))
            pseudo_supervision6 = dice_loss(image_output_soft_2_FFT, pseudo_image_output_1_FFT.unsqueeze(1))

            r = sigmoid_rampup(iter_num, 15000)

            loss_1 = (loss_dice_1 + loss_ce_1) + r* pseudo_supervision1 + r*pseudo_supervision5+ r * pseudo_supervision3
            loss_2 = (loss_dice_2 + loss_ce_2) +  r*pseudo_supervision2 + r*pseudo_supervision6+ r * pseudo_supervision4
          
            loss = loss_1 + loss_2+ r * loss_contrastive

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()

            iter_num += 1
            alpha = 0.99 + 0.01 * sigmoid_rampup(iter_num, 15000)
            update_model_ema(model_1, ema_model, alpha)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/model1_loss', loss_1, iter_num)
            writer.add_scalar('info/model2_loss', loss_2, iter_num)
            writer.add_scalar('info/model1/mix_dice', loss_dice_1, iter_num)
            writer.add_scalar('info/model1/mix_ce', loss_ce_1, iter_num)
            writer.add_scalar('info/model2/mix_dice', loss_dice_2, iter_num)
            writer.add_scalar('info/model2/mix_ce', loss_ce_2, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f' % (iter_num, loss, loss_1, loss_2))

            if iter_num > 0 and iter_num % 200 == 0:
                model_1.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model_1,
                                                  classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                performance1 = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(iter_num, round(
                        best_performance1, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model1.pth'.format(args.model))
                    torch.save(model_1.state_dict(), save_mode_path)
                    torch.save(model_1.state_dict(), save_best_path)

                logging.info('iteration %d : model1_mean_dice : %f' % (iter_num, performance1))
                model_1.train()

                model_2.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model_2,
                                                  classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                performance2 = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(iter_num, round(
                        best_performance2, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model2.pth'.format(args.model))
                    torch.save(model_2.state_dict(), save_mode_path)
                    torch.save(model_2.state_dict(), save_best_path)

                logging.info('iteration %d : model2_mean_dice : %f' % (iter_num, performance2))
                model_2.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "ABD-main/model/BCP/ACDCtrainr15000_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "ABD-main/model/BCP/ACDCtrainr15000_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    current_script = __file__  # 获取当前脚本的完整路径
    shutil.copy(current_script, snapshot_path)

    # Pre_train
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # Self_train
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)





