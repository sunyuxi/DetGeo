# -*- coding: utf8 -*-

import os
import sys
import argparse
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import gc
import cv2

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from dataset.data_loader import RSDataset
from model.DetGeo import DetGeo
from model.loss import yolo_loss, build_target, adjust_learning_rate
from utils.utils import AverageMeter, eval_iou_acc
from utils.checkpoint import save_checkpoint, load_pretrain

def main():
    parser = argparse.ArgumentParser(
        description='cross-view object geo-localization')
    parser.add_argument('--gpu', default='0,1', help='gpu id')
    parser.add_argument('--num_workers', default=24, type=int, help='num workers for data loading')

    parser.add_argument('--max_epoch', default=25, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--emb_size', default=512, type=int, help='embedding dimensions')
    parser.add_argument('--img_size', default=1024, type=int, help='image size')
    parser.add_argument('--data_root', type=str, default='./data', help='path to the root folder of all dataset')
    parser.add_argument('--data_name', default='CVOGL_DroneAerial', type=str, help='CVOGL_DroneAerial/CVOGL_SVI')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH')
    parser.add_argument('--print_freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--beta', default=1.0, type=float, help='the weight of cls loss')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--val', dest='val', default=False, action='store_true', help='val')
    
    global args, anchors_full
    args = parser.parse_args()
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)

    eps=1e-10
    ## following anchor sizes calculated by kmeans under args.anchor_imsize=1024
    if args.data_name == 'CVOGL_DroneAerial':
        anchors = '37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573'
    elif args.data_name == 'CVOGL_SVI':
        anchors = '37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573'
    else:
        assert(False)
    args.anchors = anchors

    ## save logs
    if args.savename=='default':
        args.savename = '%s_batch%d' % (args.dataset, args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename="./logs/%s"%args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv))
    logging.info(str(args))

    input_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RSDataset(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='train',
                         img_size=args.img_size,
                         transform=input_transform,
                         augment=True)
    val_dataset = RSDataset(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='val',
                         img_size = args.img_size,
                         transform=input_transform)
    test_dataset = RSDataset(data_root=args.data_root,
                         data_name=args.data_name,
                         split_name='test',
                         img_size = args.img_size,
                         transform=input_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=False, num_workers=args.num_workers)
    
    ## Model
    model = DetGeo()

    model = torch.nn.DataParallel(model).cuda()

    if args.pretrain:
        model = load_pretrain(model, args, logging)
    
    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d'%int(sum([param.nelement() for param in model.parameters()])))

    optimizer = torch.optim.RMSprop([{'params': model.parameters()},], lr=args.lr, weight_decay=0.0005)
    
    ## training and testing
    best_accu = -float('Inf')
    
    if args.test:
        _ = test_epoch(test_loader, model, args)
    elif args.val:
        _ = test_epoch(val_loader, model, args)
    else:
        for epoch in range(args.max_epoch):
            adjust_learning_rate(args, optimizer, epoch)
            gc.collect()
            train_epoch(train_loader, model, optimizer, epoch, args)
            accu_new = test_epoch(val_loader, model, args)
            ## remember best accu and save checkpoint
            is_best = accu_new > best_accu
            best_accu = max(accu_new, best_accu)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': accu_new,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args, filename=args.savename)
        print('\nBest Accu: %f\n'%best_accu)
        logging.info('\nBest Accu: %f\n'%best_accu)

def train_epoch(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter()
    avg_losses = AverageMeter()
    avg_cls_losses = AverageMeter()
    avg_geo_losses = AverageMeter()
    avg_accu = AverageMeter()
    avg_accu_center = AverageMeter()
    avg_iou = AverageMeter()

    model.train()
    end = time.time()
    anchors_full = np.array([float(x.strip()) for x in args.anchors.split(',')])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()
    
    for batch_idx, (query_imgs, rs_imgs, mat_clickxy, ori_gt_bbox, _) in enumerate(train_loader):
        query_imgs, rs_imgs = query_imgs.cuda(), rs_imgs.cuda()
        mat_clickxy = mat_clickxy.cuda()
        ori_gt_bbox = ori_gt_bbox.cuda()
        ori_gt_bbox = torch.clamp(ori_gt_bbox, min=0, max=args.img_size-1)

        pred_anchor, _ = model(query_imgs, rs_imgs, mat_clickxy)
        pred_anchor = pred_anchor.view(pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3])
        
        ## convert gt box to center+offset format
        new_gt_bbox, best_anchor_gi_gj = build_target(ori_gt_bbox, anchors_full, args.img_size, pred_anchor.shape[3])
        
        # loss
        loss_geo, loss_cls = yolo_loss(pred_anchor, new_gt_bbox, anchors_full, best_anchor_gi_gj, args.img_size)
        loss = loss_cls + loss_geo * args.beta

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_losses.update(loss.item(), query_imgs.shape[0])
        avg_geo_losses.update(loss_geo.item(), query_imgs.shape[0])
        avg_cls_losses.update(loss_cls.item(), query_imgs.shape[0])
        
        accu_list, accu_center, iou, _, _, _ = eval_iou_acc(pred_anchor, ori_gt_bbox, anchors_full, best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2], args.img_size, iou_threshold_list=[0.5])
        accu = accu_list[0]
        ## metrics
        avg_iou.update(iou, query_imgs.shape[0])
        avg_accu.update(accu, query_imgs.shape[0])
        avg_accu_center.update(accu_center, query_imgs.shape[0])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                'Geo Loss {geo.val:.4f} ({geo.avg:.4f})\t' \
                'Cls Loss {cls.val:.4f} ({cls.avg:.4f})\t' \
                'Accu {accu.val:.4f} ({accu.avg:.4f})\t' \
                'Mean_iou {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Accu_c {accu_c.val:.4f} ({accu_c.avg:.4f})\t' \
                .format( \
                    epoch, batch_idx, len(train_loader), batch_time=batch_time, \
                    loss=avg_losses, geo=avg_geo_losses, cls=avg_cls_losses, accu=avg_accu, miou=avg_iou, accu_c=avg_accu_center)
            print(print_str)
            logging.info(print_str)

def test_epoch(data_loader, model, args):
    batch_time = AverageMeter()
    avg_accu50 = AverageMeter()
    avg_accu25 = AverageMeter()
    avg_iou = AverageMeter()
    avg_accu_center = AverageMeter()
    
    torch.cuda.empty_cache()
    model.eval()
    end = time.time()
    #print(datetime.datetime.now())
    anchors_full = np.array([float(x.strip()) for x in args.anchors.split(',')])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    anchors_full = torch.tensor(anchors_full, dtype=torch.float32).cuda()

    for batch_idx, (query_imgs, rs_imgs, mat_clickxy, ori_gt_bbox, _) in enumerate(data_loader):
        query_imgs, rs_imgs = query_imgs.cuda(), rs_imgs.cuda()
        mat_clickxy = mat_clickxy.cuda()
        ori_gt_bbox = ori_gt_bbox.cuda()
        ori_gt_bbox = torch.clamp(ori_gt_bbox, min=0, max=args.img_size-1)

        with torch.no_grad():
            pred_anchor, attn_score = model(query_imgs, rs_imgs, mat_clickxy)
        pred_anchor = pred_anchor.view(pred_anchor.shape[0],\
            9, 5, pred_anchor.shape[2], pred_anchor.shape[3])
        
        _, best_anchor_gi_gj = build_target(ori_gt_bbox, anchors_full, args.img_size, pred_anchor.shape[3])
        
        accu_list, accu_center, iou, each_acc_list, _, _ = eval_iou_acc(pred_anchor, ori_gt_bbox, anchors_full, best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2], args.img_size, iou_threshold_list=[0.5, 0.25])
        
        avg_accu50.update(accu_list[0], query_imgs.shape[0])
        avg_accu25.update(accu_list[1], query_imgs.shape[0])
        avg_iou.update(iou, query_imgs.shape[0])
        avg_accu_center.update(accu_center, query_imgs.shape[0])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Accu50 {accu50.val:.4f} ({accu50.avg:.4f})\t' \
                'Accu25 {accu25.val:.4f} ({accu25.avg:.4f})\t' \
                'Mean_iou {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Accu_c {accu_c.val:.4f} ({accu_c.avg:.4f})\t' \
                .format( \
                    batch_idx, len(data_loader), batch_time=batch_time, \
                    accu50=avg_accu50, accu25=avg_accu25, miou=avg_iou, accu_c=avg_accu_center)
            print(print_str)
            logging.info(print_str)
    print(avg_accu50.avg, avg_accu25.avg, avg_iou.avg, avg_accu_center.avg)
    logging.info("%f, %f, %f, %f" % (avg_accu50.avg, avg_accu25.avg, float(avg_iou.avg), avg_accu_center.avg))

    return avg_accu50.avg


if __name__ == "__main__":
    main()
