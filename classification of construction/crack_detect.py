import sys
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from unet.unet_transfer import UNet16, input_size
import matplotlib.pyplot as plt
import argparse
from os.path import join
from PIL import Image
import gc
from utils import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34
from tqdm import tqdm
import cv2
# from roboflow import Roboflow
# from pathlib import Path

# rf = Roboflow(api_key="WrFcgRAE0jnhmUqmM7nB")
# project = rf.workspace().project("crack-dbrlf")
# model1 = project.version(1).model

def evaluate_img(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
    X = train_tfms(Image.fromarray(img_1))
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

    mask = model(X)

    mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
    return mask

def evaluate_img_patch(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_height, img_width, img_channels = img.shape

    if img_width < input_width or img_height < input_height:
        return evaluate_img(model, img)

    stride_ratio = 0.1
    stride = int(input_width * stride_ratio)

    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []
    for y in range(0, img_height - input_height + 1, stride):
        for x in range(0, img_width - input_width + 1, stride):
            segment = img[y:y + input_height, x:x + input_width]
            # print(y,y+input_height,x,x+input_width)
            # print(img[y:y + input_height, x:x + input_width])
            # img = cv2.rectangle(img,(y,y+input_height),(x,x+input_width),(255,0,0),3)
           
            
            normalization_map[y:y + input_height, x:x + input_width] += 1
            patches.append(segment)
            patch_locs.append((x, y))

    patches = np.array(patches)
    if len(patch_locs) <= 0:
        return None

    preds = []
    for i, patch in enumerate(patches):
        patch_n = train_tfms(Image.fromarray(patch))
        X = Variable(patch_n.unsqueeze(0)).cuda()  # [N, 1, H, W]
        masks_pred = model(X)
        mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        preds.append(mask)

    probability_map = np.zeros((img_height, img_width), dtype=float)
    for i, response in enumerate(preds):
        coords = patch_locs[i]
        probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response
    
    return probability_map

def disable_axis():
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-img_dir',type=str, help='input dataset directory')
    # # parser.add_argument('-model_path', type=str, help='trained model path')
    # parser.add_argument('-model_type', type=str, choices=['vgg16', 'resnet101', 'resnet34'])
    # parser.add_argument('-out_viz_dir', type=str, default='', required=False, help='visualization output dir')
    # parser.add_argument('-out_pred_dir', type=str, default='', required=False,  help='prediction output dir')
    parser.add_argument('-threshold', type=float, default=0.2 , help='threshold to cut off crack response')
    args = parser.parse_args()

    if Path("./viz") != '':
        os.makedirs(Path("./viz"), exist_ok=True)
        for path in Path("./viz").glob('*.*'):
            os.remove(str(path))

    if Path("./result") != '':
        os.makedirs(Path("./result"), exist_ok=True)
        for path in Path("./result").glob('*.*'):
            os.remove(str(path))

    # if args.model_type == 'vgg16':
    model = load_unet_vgg16(Path("model_unet_vgg_16_best.pt"))
    # elif args.model_type  == 'resnet101':
    #     model = load_unet_resnet_101(args.model_path)
    # elif args.model_type  == 'resnet34':
    #     model = load_unet_resnet_34(args.model_path)
    #     print(model)
    # else:
    #     print('undefind model name pattern')
    #     exit()

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    # 사진 불러오는 공간 설정

    # paths = [path for path in Path("./crack_image").glob('*.*')]
    paths = [path for path in Path("./crack_image").glob('*.*')]



    for path in tqdm(paths):
        #print(str(path))

        train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

        img_0 = Image.open(str(path))
        img_0 = np.asarray(img_0)
        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            continue

        img_0 = img_0[:,:,:3]

        img_height, img_width, img_channels = img_0.shape

        prob_map_full = evaluate_img(model, img_0)


        if Path("./result") != '':
            cv.imwrite(filename=join(Path("./result"), f'{path.stem}.jpg'), img=(prob_map_full * 255).astype(np.uint8))

        if Path("./viz") != '':
            # plt.subplot(121)
            # plt.imshow(img_0), plt.title(f'{img_0.shape}')
            if img_0.shape[0] > 2000 or img_0.shape[1] > 2000:
                img_1 = cv.resize(img_0, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)
            else:
                img_1 = img_0

            # plt.subplot(122)
            # plt.imshow(img_0), plt.title(f'{img_0.shape}')
            # plt.show()

            prob_map_patch = evaluate_img_patch(model, img_1)

            #plt.title(f'name={path.stem}. \n cut-off threshold = {args.threshold}', fontsize=4)
            prob_map_viz_patch = prob_map_patch.copy()
            prob_map_viz_patch = prob_map_viz_patch/ prob_map_viz_patch.max()
            prob_map_viz_patch[prob_map_viz_patch < args.threshold] = 0.0

            

            cv.imwrite(filename=join(Path("./crack_edge"), "mask_opencv "+f'{path.stem}.jpg'), img=(prob_map_viz_patch * 255).astype(np.uint8))

            # model1.predict("crack_edge/mask_opencv "+f'{path.stem}.jpg', confidence=40, overlap=30).save("prediction"+f'{path.stem}.jpg')

            # img = cv2.imread("prediction"+f'{path.stem}.jpg')

            
            fig = plt.figure()
            # st = fig.suptitle(f'name={path.stem} \n cut-off threshold = {args.threshold}', fontsize="x-large")
            ax = fig.add_subplot(131)
            ax.imshow(img_1)
            ax = fig.add_subplot(132)
            ax.imshow(prob_map_viz_patch)
            ax = fig.add_subplot(133)
            ax.imshow(img_1)
            # ax.imshow(img)

            # img3=cv2.imread(join(Path("./crack_edge"), "mask_opencv "+f'{path.stem}.jpg'),cv2.IMREAD_GRAYSCALE)
            # blurred=cv2.GaussianBlur(img3,(0,0),1)

            # th1=100
            # th2=200
            # no_blur=cv2.Canny(img3,th1,th2)
            # yes_blur=cv2.Canny(blurred,th1,th2)

            # cv2.imshow("canny_img", yes_blur)
            # cv2.waitKey(0)

            ax.imshow(prob_map_viz_patch, alpha=0.4)
            # ax.imshow(yes_blur, alpha=0.4)

            plt.savefig(join(Path("./viz"), f'{path.stem}.jpg'), dpi=500)
            plt.close('all')

            # cv2.namedWindow("edgeadd")
            # cv2.imshow("edgeadd",img_1)
            # dst1 = cv2.add(img_1, prob_map_viz_patch, dtype=cv2.CV_8U)
            # cv2.imshow("edgeadd",prob_map_viz_patch, alpha = 0.4)
            # cv2.imshow("edgeadd",dst1)
            # cv2.waitKey(0)


            
            fig1 = plt.figure()
            ax1 = fig1.add_subplot()
            ax1.axis('off')
            ax1.imshow(img_1)
            ax1.imshow(prob_map_viz_patch, alpha=0.4)

            # cv2.imshow("",prob_map_viz_patch)

            # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            # print(prob_map_viz_patch)
            
            
            plt.savefig(join(Path("./crack_edge"), 'mask' + f'{path.stem}.jpg'), dpi=500, bbox_inches='tight', pad_inches=0)
            plt.close('all')

        

            # image = cv2.imread("./crack_edge/mask_opencv"+f'{path.stem}.jpg', cv2.IMREAD_GRAYSCALE)

            # def onChange():
            #     pass    

            # cv2.namedWindow("result")
            # a = cv2.getTrackbarPos('A', 'result')
            # cv2.createTrackbar('A', 'result', 0, 255, a)

            # cv2.createTrackbar('B', 'result', 0, 255, onChange)
            # b = cv2.getTrackbarPos('B', 'result')


            

            # cv2.imshow('',prob_map_viz_patch)
            # cv2.waitKey(0)

            # prob_map_viz_full = prob_map_full.copy()
            # prob_map_viz_full[prob_map_viz_full < args.threshold] = 0.0
            

            # ax = fig.add_subplot(234)
            # ax.imshow(img_0)
            # ax = fig.add_subplot(235)
            # ax.imshow(prob_map_viz_full)
            # ax = fig.add_subplot(236)
            # ax.imshow(img_0)
            # ax.imshow(prob_map_viz_full, alpha=0.4)

            

        gc.collect()