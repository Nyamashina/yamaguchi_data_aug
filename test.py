import pandas as pd
import numpy as np
import path
import json
import os
import pathlib
import os.path as osp
import sys
import random
import time
import glob
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
import glob
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from torch.utils.data import Dataset, DataLoader
import gc
from timm.data import ImageDataset, create_loader
from functools import lru_cache

import ffmpeg


err_tol = {
    'challenge': [ 0.30, 0.40, 0.50, 0.60, 0.70 ],
    'play': [ 0.15, 0.20, 0.25, 0.30, 0.35 ],
    'throwin': [ 0.15, 0.20, 0.25, 0.30, 0.35 ]
}
video_id_split = {
    'val':[
         '3c993bd2_0',
         '3c993bd2_1',
    ],
    'train':[
         '1606b0e6_0',
         '1606b0e6_1',
         '35bd9041_0',
         '35bd9041_1',
         '407c5a9e_1',
         '4ffd5986_0',
         '9a97dae4_1',
         'cfbe2e94_0',
         'cfbe2e94_1',
         'ecf251d4_0',
    ]
}
event_names = ['challenge', 'throwin', 'play']
label_dict = {
    'background':0,
    'challenge':1,
    'play':2,
    'throwin':3,
}
event_names_with_background = ['background','challenge','play','throwin']


def make_sub(prob, filenames):
    frame_rate = 25
    ignore_width = 14

    df = pd.DataFrame(prob, columns=event_names_with_background)
    df['video_name'] = filenames
    df['video_id'] = df['video_name'].apply(lambda x: '-'.join(x.split('-')[:-1]).split('/')[-1])
    df['frame_id'] = df['video_name'].str.split('-').str[1].str.split('.').str[0].astype(int)

    train_df = []
    for video_id, gdf in df.groupby('video_id'):
        for i, event in enumerate(event_names):
            # print(video_id, event)
            prob_arr = gdf[event].values
            sort_arr = np.argsort(-prob_arr)
            rank_arr = np.empty_like(sort_arr)
            rank_arr[sort_arr] = np.arange(len(sort_arr))
            idx_list = []
            for i in range(len(prob_arr)):
                this_idx = sort_arr[i]
                if this_idx >= 0:
                    idx_list.append(this_idx)
                    for parity in (-1, 1):
                        for j in range(1, ignore_width + 1):
                            ex_idx = this_idx + j * parity
                            if ex_idx >= 0 and ex_idx < len(prob_arr):
                                sort_arr[rank_arr[ex_idx]] = -1
            this_df = gdf.reset_index(drop=True).iloc[idx_list].reset_index(drop=True)
            this_df["score"] = prob_arr[idx_list]
            this_df['event'] = event
            train_df.append(this_df)
    train_df = pd.concat(train_df)
    train_df['time'] = train_df['frame_id'] / frame_rate

    return train_df.reset_index(drop=True)
#----------------------------------------------------
VALID = True # for validation
TEST = True # for submission
torch.backends.cudnn.benchmark = True
if TEST:
    test_df = make_sub(prob_test, filenames_valid)
    test_df[['video_id', 'time', 'event', 'score']].to_csv("submission.csv", index=False)

path = 'train/'
json_file = glob.glob(path + '*.json')
json_file = [os.path.basename(json_f) for json_f in json_file]
mp4_file = glob.glob(path + '*.mp4')
mp4_file = [os.path.basename(mp4) for mp4 in mp4_file]
#----------------------------------------------------



if TEST:
    OUT_DIR = '/kaggle/work/extracted_images_test'
    IN_DIR = '../input/dfl-bundesliga-data-shootout/test'
    IN_VIDEOS = sorted(glob.glob('../input/dfl-bundesliga-data-shootout/test/*'))

filenames_valid = glob.glob('/kaggle/work/extracted_images_test/*')
print(len(filenames_valid))

tmp_df = pd.DataFrame({'video_name': filenames_valid})
tmp_df['video_id'] = tmp_df['video_name'].apply(lambda x:'-'.join(x.split('-')[:-1]).split('/')[-1])
tmp_df['frame_id'] = tmp_df['video_name'].str.split('-').str[-1].str.split('.').str[0].astype(int)
tmp_df = tmp_df.sort_values(by=['video_id', 'frame_id']).reset_index(drop=True)
filenames_valid = tmp_df['video_name'].values.tolist()


#=====================================================================================================================
def load_model(model, path, classes=4):
    model = timm.create_model(model, pretrained=False, num_classes=classes, in_chans=9).to('cuda')
    model.load_state_dict(torch.load(f'{path}'))
    model.eval()
    return model
#=====================================================================================================================

#=====================================================================================================================
def movie_to_images(mp4,SAVE_MODE=False,RESIZE = True,image_width = 1920,image_height = 1080,laplacian_thr = 0):
    print(mp4)
    mp4_data = str(path+mp4)
    Image_dict = dict()
    i = 0
    count = 0
    cpf = 1
    cap = cv2.VideoCapture(mp4_data)
    while (cap.isOpened()):
        ret, frame = cap.read()  # 動画を読み込む
        # assert frame, "オープンに失敗"            #デバッグ用

        if ret == False:
            print('Finished')  # 動画の切り出しが終了した時
            break

        if count % cpf == 0:  # 何フレームに１回切り出すか
            if RESIZE:            # サイズを小さくする
                 frame = cv2.resize(frame, (image_width, image_height))

            # 画像がぶれていないか確認する
            laplacian = cv2.Laplacian(frame, cv2.CV_64F)
            if ret and laplacian.var() >= laplacian_thr:  # ピンぼけ判定がしきい値以上のもののみ出力

                # 第１引数画像のファイル名、第２引数保存したい画像
                if SAVE_MODE:
                    write = cv2.imwrite(f'./images/{mp4}_i{i}.png',frame)  # 切り出した画像を表示する
                    assert write, "保存に失敗"
                Image_dict[i] = frame
                print(f'{mp4}_{i}...clear')  # 確認用表示
                i += 1

        count = count + 1

    cap.release()
    return Image_dict

#=====================================================================================================================
def extract_images(video_path, out_dir):
    video_name = os.path.basename(video_path).split('.')[0]
    print(video_path)
    """
    ffmpeg \
    (グローバルオプション) \
    (入力1オプション) -i (入力ソース1) \
    (入力2オプション) -i (入力ソース2) \
    ... \
    (出力1オプション) (出力先1) \
    (出力2オプション) (出力先2) \
    ...
    """
    #!ffmpeg -y -c:v h264_cuvid -i {video_path} -q:v 2 -s 512*512 -f image2 {out_dir}/{video_name} -%06d.jpg -hide_banner -loglevel error

    gc.collect()


test_aug = A.Compose([A.Resize(512, 512),A.Normalize(mean=[0.], std=[1.]),ToTensorV2()])

#=====================================================================================================================
class MyDataSet(Dataset):
    def __init__(self,path,aug=test_aug,mode='train'):
        self.path = path
        self.aug = aug
        self.mode = mode

    @lru_cache(128)
    def read_img(selfself,video_id,tmp_idx):
        return cv2.imread(f'/kaggle/work/extracted_images_test/{video_id}-{tmp_idx:06d}.jpg')

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        filename = self.path[idx]

        if self.mode == 'train':
            tmp_id = filename.split('/')[-1].split('_')
            video_id = tmp_id[0] + '_' + tmp_id[1]
            frame_idx = int(tmp_id[-1].split('.')[0])
        else:
            tmp_id = filename.split('/')[-1].split('-')
            video_id = tmp_id[0]
            frame_idx = int(tmp_id[-1].split('.')[0])

        frames = []

        for tmp_idx in range(frame_idx-4,frame_idx+5):
            img = self.read_img(video_id,frame_idx)

            if img is None:
                img = self.read_img(video_id,frame_idx)

            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            frames.append(img)

        frames = np.array(frames).transpose(1,2,0)
        frames = self.aug(image=frames)["image"]

        if self.mode == 'train':
            label = filename.split('/')[-2]
            label = ['background', 'challenge', 'play', 'throwin'].index(label)
        else:
            label = -1

        return frames, label

#=====================================================================================================================
for video_path in IN_VIDEOS:
    extract_images(video_path, OUT_DIR)

#=====================================================================================================================
@torch.no_grad()
def inference(models1, models2, models3, filenames_valid):
    stride = len(models1)

    test_set = MyDataset(filenames_valid, test_aug, 'test')
    test_loader = DataLoader(test_set, batch_size=72, shuffle=False, num_workers=2)

    probs1 = [[] for i in range(stride)]
    probs2 = [[] for i in range(stride)]
    probs3 = [[] for i in range(stride)]

    tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
    for step, batch in enumerate(tk):
        img, _ = [x.to('cuda', non_blocking=True) for x in batch]

        for i in range(stride):
            labels = models1[i](img[i::stride])
            probs1[i].append(torch.softmax(labels, -1))
            labels = models2[3 - i](img[i::stride])
            probs2[i].append(torch.softmax(labels, -1))
            labels = models3[i](img[i + 1::6]).view(-1, 4)
            probs3[i].append(torch.softmax(labels, -1))

    for i in range(stride):
        probs1[i] = torch.cat(probs1[i]).cpu().numpy()
        probs2[i] = torch.cat(probs2[i]).cpu().numpy()
        probs3[i] = torch.cat(probs3[i]).cpu().numpy()
    return probs1, probs2, probs3, filenames_valid
#=====================================================================================================================

filenames_valid = filenames_valid[:len(filenames_valid) - len(filenames_valid) % 12]

models1 = []
for i in range(4):
    models1.append(load_model('swsl_resnext50_32x4d', f'../input/dfl-resnext50-4fold/model_fold{i}.pt'))

models2 = []
for i in range(4):
    models2.append(load_model('swsl_resnext101_32x4d', f'../input/resnext101-4fold/model_fold{i}.pt'))

models3 = []
for i in range(4):
    models3.append(load_model('swsl_resnext50_32x4d', f'../input/dfl-resnext50-frames3-4fold/model_fold{i}.pt', 4 * 3))

probs1, probs2, probs3, _ = inference(models1, models2, models3, filenames_valid)

#=====================================================================================================================
def labeling_i(jsonpath):
    jsonpath = path+jsonpath
    label1_ng = list()
    label1_ok = list()
    label2_ng = list()
    label2_ok = list()
    label3_ng = list()
    label3_ok = list()
    jsonfile = open(jsonpath, 'r', encoding="utf-8")
    print('---------------------------------------------------------')
    print(jsonfile,jsonpath)
    print('---------------------------------------------------------')
    label_all = json.load(jsonfile)

    labels = list(label_all[N].pop("labels") for N in range(len(label_all)))
    frame_id = list(label_all[N].pop("frame_id") for N in range(len(label_all)))
    tot_length = len(frame_id)
    for N in range(len(labels)):                                               
        for item in labels[N].items():
            print(item)
            if "要補修-1.区画線" in item:
                label1_ng += [N,item]
                print('label1_ng')

            if "補修不要-1.区画線" in item:
                label1_ok += [N,item]
                print('label1_ok')

            if "要補修-2.道路標識" in item:
                label2_ng += [N,item]
                print('label2_ng')

            if "補修不要-2.道路標識" in item:
                label2_ok += [N,item]
                print('label2_ok')

            if "要補修-3.照明" in item:
                label3_ng += [N,item]
                print('label3_ng')

            if "補修不要-3.照明" in item:
                label3_ok += [N,item]
                print('label3_ok')

    print('---------------------------------------------------------')
    print('label1_ng\n', label1_ng)
    print('label1_ok\n', label1_ok)
    print('label2_ng\n', label2_ng)
    print('label2_ok\n', label2_ok)
    print('label3_ng\n',label3_ng)
    print('label3_ok\n',label3_ok)
    print('---------------------------------------------------------')
    json_ith_data = [label1_ng,label1_ok,label2_ng,label2_ok,label3_ng,label3_ok]

    return tot_length,json_ith_data
#=====================================================================================================================



print(json_file)
train_img_list = []
#リスト内はｊ番目にシーンjの動画dict,dictは引数iの要素にi秒目の画像を対応させている
train_json = []
for json_f in json_file:
    print(json_f)
    length, data = labeling_i(json_f)
    train_json += [length, data]

for mp4 in mp4_file:
    images = movie_to_images(mp4)
    train_img_list += images

print(train_json)


voc_classes = ['label1_ng','label1_ok', 'label2_ng','label2_ok','label3_ng','label3_ok']


(train_df if VALID else test_df)[['video_id', 'time', 'event', 'score']]