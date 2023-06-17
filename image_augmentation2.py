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
import Image
import glob
import os
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


#----------------------------------------------------
Imagefolderpath = './train_data/'
if not os.path.exists(Imagefolderpath):
    os.makedirs(Imagefolderpath)

train_path = 'train/'
test_path = 'test/'
json_file = glob.glob(train_path + '*.json')
json_file = [os.path.basename(json_f) for json_f in json_file]
mp4_file = glob.glob(train_path + '*.mp4')
mp4_file = [os.path.basename(mp4) for mp4 in mp4_file]
test_mp4_file = glob.glob(test_path + '*.mp4')
test_mp4_file = [os.path.basename(mp4) for mp4 in test_mp4_file]


print(json_file)
print(mp4_file)
#----------------------------------------------------
class Data:
    horizontal_transform = A.Compose([
        A.HorizontalFlip(p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    randomsizedcrop_transform = A.Compose([
        A.RandomSizedCrop(min_max_height=[512, 512], height=1024, width=1024, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    rotate90_transform = A.Compose([
        A.RandomRotate90(p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    rotate180_transform = A.Compose([
        A.RandomRotate90(p=1.0),
        A.RandomRotate90(p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    rotate270_transform = A.Compose([
        A.RandomRotate90(p=1.0),
        A.RandomRotate90(p=1.0),
        A.RandomRotate90(p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __init__(self, image="", bboxes=0, id="", class_labels=[]):
        self.image = image
        self.bboxes = bboxes
        self.label = 0
        self.id = id
        self.class_labels = class_labels

    # jpgとtxtからデータをインポートするメソッド
    def importdata(self, imgpath):
        dirpath = os.path.dirname(imgpath)[:-7]
        id = os.path.splitext(os.path.basename(imgpath))[0]
        txtpath = dirpath + f"/labels/{id}.txt"

        img = cv2.imread(imgpath)
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = []
        with open(txtpath) as f:
            for line in f:
                line_list = line.split(" ")
                bbox = line_list[1:]
                bbox = [float(i.replace('\n', '')) for i in bbox]
                bboxes.append(bbox)

        self.bboxes = bboxes
        self.label = 0
        self.id = id
        self.class_labels = ["wheat" for i in range(len(bboxes))]

    # albumentationsで変換したデータをインポートするメソッド
    def import_transformdata(self, transform_data, origin_data, process):
        self.image = transform_data["image"]
        self.bboxes = transform_data["bboxes"]
        self.label = 0
        self.id = origin_data.id + "_" + process
        self.class_labels = transform_data["class_labels"]

    # モザイク画像のデータをインポートするメソッド
    def import_mosaicdata(self, img, bboxes, id, class_labels):
        self.image = img
        self.bboxes = bboxes
        self.label = 0
        self.id = id
        self.class_labels = class_labels

    # 左右反転処理したデータを返すメソッド
    def horizonflip(self):
        horizon_transformed = Data.horizontal_transform(image=self.image,
                                                        bboxes=self.bboxes,
                                                        class_labels=self.class_labels)
        image = horizon_transformed["image"]
        bboxes = horizon_transformed["bboxes"]
        label = 0
        id = self.id + "_hori"
        class_labels = horizon_transformed["class_labels"]
        horizondata = Data(image, bboxes, id, class_labels)
        return horizondata

    # ランダムに切り出してリサイズしたデータを返すメソッド
    def randomsizedcrop(self):
        randomsizedcrop_transformed = Data.randomsizedcrop_transform(image=self.image,
                                                                     bboxes=self.bboxes,
                                                                     class_labels=self.class_labels)
        image = randomsizedcrop_transformed["image"]
        bboxes = randomsizedcrop_transformed["bboxes"]
        label = 0
        id = self.id + "_hori"
        class_labels = randomsizedcrop_transformed["class_labels"]
        randomsizedcropdata = Data(image, bboxes, id, class_labels)
        return randomsizedcropdata

    # 反時計回りに90ﾟ回転したデータを返すメソッド
    def rotate90(self):
        rotate90_transformed = Data.rotate90_transform(image=self.image,
                                                       bboxes=self.bboxes,
                                                       class_labels=self.class_labels)
        image = rotate90_transformed["image"]
        bboxes = rotate90_transformed["bboxes"]
        label = 0
        id = self.id + "_rot90"
        class_labels = rotate90_transformed["class_labels"]
        rot90data = Data(image, bboxes, id, class_labels)
        return rot90data

    # 反時計回りに180ﾟ回転したデータを返すメソッド
    def rotate180(self):
        rotate180_transformed = Data.rotate180_transform(image=self.image,
                                                         bboxes=self.bboxes,
                                                         class_labels=self.class_labels)
        image = rotate180_transformed["image"]
        bboxes = rotate180_transformed["bboxes"]
        label = 0
        id = self.id + "_rot180"
        class_labels = rotate180_transformed["class_labels"]
        rot180data = Data(image, bboxes, id, class_labels)
        return rot180data

    # 反時計回りに270ﾟ回転したデータを返すメソッド
    def rotate270(self):
        rotate270_transformed = Data.rotate270_transform(image=self.image,
                                                         bboxes=self.bboxes,
                                                         class_labels=self.class_labels)
        image = rotate270_transformed["image"]
        bboxes = rotate270_transformed["bboxes"]
        label = 0
        id = self.id + "_rot270"
        class_labels = rotate270_transformed["class_labels"]
        rot270data = Data(image, bboxes, id, class_labels)
        return rot270data

    # 指定のパスにjpgとtxtファイルでデータ保存するメソッド
    def export_data(self, imgdirpath):
        id = self.id
        dirpath = imgdirpath[:-7]
        export_imgpath = imgdirpath + f"/{id}.jpg"
        export_txtpath = dirpath + f"/labels/{id}.txt"

        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(export_imgpath, img)

        txt = ""
        for bbox in self.bboxes:
            x_min, y_min, width, height = [i for i in bbox]
            line = f"0 {x_min} {y_min} {width} {height}"
            txt += line + "\n"

        f = open(export_txtpath, 'w')
        f.write(txt)
        f.close()

    # 画像とバウンディングボックスを表示するメソッド
    def visualize(self, img_width, img_height, figsize=(10, 10)):

        for bbox in self.bboxes:
            x_mid_nor, y_mid_nor, width_nor, height_nor = [float(i) for i in bbox]

            width = width_nor * img_width
            height = height_nor * img_height

            x_min = x_mid_nor * img_width - width / 2
            y_min = y_mid_nor * img_height - height / 2
            x_max = x_min + width
            y_max = y_min + height

            x_min = int(x_min)
            x_max = int(x_max)
            y_min = int(y_min)
            y_max = int(y_max)

            img = cv2.rectangle(self.image,
                                pt1=(x_min, y_min),
                                pt2=(x_max, y_max),
                                color=(255, 0, 0),
                                thickness=3)

        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(img)


def movie_to_images(mp4,path=train_path,SAVE_MODE=True,RESIZE = True,image_width = 1920,image_height = 1080,laplacian_thr = 0):
    print(mp4)
    mp4_data = str(path+mp4)
    Image_dict = dict()
    i = 0
    count = 0
    cpf = 1
    cap = cv2.VideoCapture(mp4_data)
    while (cap.isOpened()):
        ret, frame = cap.read()  # 動画を読み込む
        assert frame, "オープンに失敗"            #デバッグ用
        print(frame)

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
                    mp4name = mp4.strip('.mp4')
                    write = cv2.imwrite(f'./images/{mp4name}_i{i}.png',frame)  # 切り出した画像を表示する
                    assert write, "保存に失敗"
                Image_dict[i] = frame
                print(f'{mp4name}_{i}...clear')  # 確認用表示
                i += 1

        count = count + 1

    cap.release()
    return Image_dict


train_img_list = []
train_json = []



def augmentation_CLAHE(image_path,PLOT = False,SAVE_MODE = True):
    CLAHE_image = cv2.imread(image_path)
    imagename = image_path.strip('\/.')
    CLAHE_image = A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=1)
    if SAVE_MODE:
        Imagefolderpath = 'train_data/'
        if not os.path.exists(Imagefolderpath):
            os.makedirs(Imagefolderpath)
        write = cv2.imwrite(f'./train_data/image_{imagename}.jpg', CLAHE_image)  # 切り出した画像を表示する
        assert write, "保存に失敗"
    if PLOT:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].imshow(CLAHE_image)
    return CLAHE_image

def augmentation_IAASharpen(image_path,PLOT = False,SAVE_MODE = False):
    IAASharpen_image = cv2.imread(image_path)
    imagename = image_path.strip('\/.')
    IAASharpen_image = A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1)
    if SAVE_MODE:
        Imagefolderpath = 'train_data/'
        if not os.path.exists(Imagefolderpath):
            os.makedirs(Imagefolderpath)
        write = cv2.imwrite(f'./train_data/image_{image_path}.jpg', IAASharpen_image)  # 切り出した画像を表示する
        assert write, "保存に失敗"
    if PLOT:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].imshow(IAASharpen_image)
    return IAASharpen_image

def augmentation_weather(image_path,weather='sunny',PLOT = False,SAVE_MODE = False):

    weather_image = cv2.imread(image_path)
    print(weather_image.shape)
    imagename = image_path.strip('train_data/')

    print(imagename)
    print(f'image_path:{image_path}')
    if weather == 'sunny':
        weather_image = A.RandomSunFlare(p=1)
    if weather == 'fog':
        weather_image = A.RandomFog(p=1)
    if weather == 'snow':
        weather_image = A.RandomSnow(p=1)
    if weather == 'rain':
        weather_image = A.RandomRain(p=1)
    if weather == 'cloudy':
        weather_image = A.RandomShadow(p=1)

    print(weather_image.shape)
    if SAVE_MODE:
        Imagefolderpath = 'train_data/'
        if not os.path.exists(Imagefolderpath):
            os.makedirs(Imagefolderpath)
        weather_image.save(f'train_data/{weather}_{imagename}')

        write = cv2.imwrite(f'train_data/{weather}_{imagename}', weather_image)  # 切り出した画像を表示する
        assert write, "保存に失敗"

    if PLOT:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].imshow(weather_image)


MOVIETOIMAGE = False
if MOVIETOIMAGE:
    for mp4 in mp4_file:
        images = movie_to_images(mp4,SAVE_MODE=True)
        print(images)
Imagefolderpath = 'train_data/'
if not os.path.exists(Imagefolderpath):
    os.makedirs(Imagefolderpath)

AUGMENTATION = True
if AUGMENTATION:
    image_file = glob.glob(Imagefolderpath + '*.jpg')
    print('images:',image_file)
    image_file = [os.path.basename(image_f) for image_f in image_file]
    print('images:', image_file)
    for image_path in image_file:
        print('Image Augmentation',image_path)
        #fog_image = augmentation_weather(Imagefolderpath + image_path,weather='fog', PLOT=False, SAVE_MODE=True)
        snow_image = augmentation_weather(Imagefolderpath + image_path, weather='snow',PLOT = False, SAVE_MODE = True)
        rain_image = augmentation_weather(Imagefolderpath + image_path, weather='rain', PLOT=False, SAVE_MODE=True)
        cloudy_image = augmentation_weather(Imagefolderpath + image_path, weather='cloudy', PLOT=False, SAVE_MODE=True)
        sunny_image = augmentation_weather(Imagefolderpath + image_path,weather='sunny', PLOT=False, SAVE_MODE=True)
        CLAHE_image = augmentation_CLAHE(Imagefolderpath + image_path,PLOT=False, SAVE_MODE=True)
        IAASharpen_image = augmentation_IAASharpen(Imagefolderpath + image_path,PLOT=False, SAVE_MODE=True)

"""
for mp4 in test_mp4_file:
    images = movie_to_images(mp4)

for json_f in json_file:
    with open(train_path+json_f, 'r', encoding="utf-8") as f:
        jsonfile = json.load(f)
    print(jsonfile)
    for frame_id in range(len(jsonfile)):
        print(frame_id,jsonfile[frame_id])
        labels = jsonfile[frame_id]['labels']
"""


