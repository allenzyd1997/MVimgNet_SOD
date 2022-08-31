
from curses import keyname
from email import iterators
from importlib.resources import path
import os
from pickletools import optimize
from textwrap import wrap 
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


import pathlib

from typing import List, Union
import PIL.Image
import torch
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn 
import numpy as np 
from pathlib import Path

from carvekit.ml.arch.u2net.u2net import U2NETArchitecture
from carvekit.ml.files.models_loc import u2net_full_pretrained
from carvekit.utils.image_utils import load_image, convert_image
from carvekit.utils.pool_utils import thread_pool_processing, batch_generator
from u2net_yidan import U2NET

from utils import wrap_yidan
from avm import AverageMeter
from dataset import SequenceDataset
# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))
	return loss0, loss



def batch_generator(iterable, n=30):
    """
        Splits any iterable into n-size packets

        Args:
            iterable: iterator
            n: size of packets

        Returns:
            new n-size packet
    """
    it = len(iterable)
    for ndx in range(0, it, n):
        yield iterable[ndx:min(ndx + n, it)]

def wrap_flow(batch, src_0):
    K = 3 
    # should be check for the global parameter
    flow_root = Path('/data_sobig/zhangyidan/flows')
    vd_src = Path(src_0)
    # start_pic = int(vd_src.name[:3])
    start_pic = vd_src.name
    start_pic_num = int(vd_src.name[:3])
    vd_flow_cls = Path(vd_src.parents[1].name)
    vd_flow_name =  Path(vd_src.parent.name)
    vd_flow_root = flow_root/vd_flow_cls/vd_flow_name/Path("of_03.lst.npy")

    vd_flow_dict = np.load(vd_flow_root, allow_pickle=True).item()
    vd_flow_keys = list(vd_flow_dict.keys())
    start_pic_pl = vd_flow_keys.index(start_pic)
    wrap_batch = [] 
    
    for idx, x in enumerate(batch):
        # keyname = "{pn:03d}.jpg".format(pn=start_pic + idx)
        keyname = vd_flow_keys[start_pic_pl + idx]
        # keyname = "001.jpg"
        try:
            flow_val = vd_flow_dict[keyname]
        except:
            filew = open('/data/zhangyidan/video_sod/image-background-remove-tool/carvekit/ml/train/recordfiles/err.log', 'w')
            print("keyname: " + str(keyname))
            print("vd_flow_root: " )
            print(vd_flow_root)
            filew.write("keyname: " + str(keyname) +'\n')
            filew.write("vd_flow_root: " + '\n')
            filew.write(str(vd_flow_root))
            filew.close()

        if start_pic_num + idx <= K:
            npx = batch[idx + K].cpu().data.numpy() 
        else:
            npx = batch[idx - K].cpu().data.numpy() 
            flow_val = -1 * flow_val

        npx = npx.reshape((320,320,1))
        npx = wrap_yidan(npx, flow_val)
        npx = npx.reshape((1,320,320))
        wrap_batch.append(npx)
    wrap_batch = np.array(wrap_batch)
    return wrap_batch

def train(model,data_set, epoch = 30 , batch_size = 30):
    
    
    it_total = len(list(data_set.files))//30
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    for ep_n in range(epoch):
        loss_avm = AverageMeter()
        loss0_avm = AverageMeter()
        print("EPOCH: {epoch:3d} Begin to Train".format(epoch = ep_n))
        batches = batch_generator(data_set.files, n = 30)
        p_bar = tqdm(batches)
        
        for idx, batch in enumerate(p_bar):
            batch_path = [x[0] for x in batch[:10]]

            d0, d1, d2, d3, d4, d5, d6 = model(batch_path)
            # out is the PIL list    
            wrap_out = wrap_flow(d0, batch_path[0])
            wrap_out = torch.from_numpy(wrap_out).cuda()
            loss0, loss = muti_loss_fusion(d0, d1, d2, d3, d4, d5, d6, wrap_out)
            
            loss_avm.update(loss.item())
            loss0_avm.update(loss0.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            info_up = "Train Epoch: {epoch}/{epochs:4}. Iteration: {iteration:4}/{iterations:4}, loss: {loss_avg:.4f}, loss0: {loss0_avg:.4f}".format(
                epoch = ep_n,
                epochs = epoch,
                iteration =idx,
                iterations = it_total,
                loss_avg = loss_avm.avg,
                loss0_avg = loss0_avm.avg,
            )
            p_bar.set_description(info_up)
            p_bar.update()

        if ep_n % 5 == 0 :
            torch.save(model.state_dict(), "/data/zhangyidan/video_sod/image-background-remove-tool/carvekit/ml/train/recordfiles/ft_flow_{ep_n:02d}.pth".format(ep_n=ep_n))
        p_bar.close()
    return model 



if __name__ == "__main__":
    
    pth0 = '/data1/zhangyidan/mvImgNet'
    pths = [pth0]
    sd = SequenceDataset(paths=pths)

    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    devices_id = [0,1,2]
    # print(len(sd.files))
    model = U2NET(device = device, batch_size = 10)
    model = torch.nn.DataParallel(model , devices_id)
    
    # torch.save(model.state_dict(), "/data/zhangyidan/video_sod/image-background-remove-tool/carvekit/ml/train/recordfiles/ft_flow.pth")
    model = train(model, sd)
    torch.save(model.state_dict(), "/data/zhangyidan/video_sod/image-background-remove-tool/carvekit/ml/train/recordfiles/ft_flow.pth")
    print('end')
