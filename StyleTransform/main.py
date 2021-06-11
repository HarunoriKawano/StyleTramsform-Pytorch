"""
StyleTransformByPytorch
python version 3.8
pytorch version 1.7.1+cu110
"""

import numpy
import torch
from torchvision import models
import numpy as np
from models import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('run on ' + device)

    content_path = '../sample_images/coffee.jpg'
    style_path = '../sample_images/sea.jpg'

    content_img = image_loader(content_path, gray_scale=True, device=device)
    style_img = image_loader(style_path, gray_scale=False, device=device)
    image_show(content_img, title='content')
    image_show(style_img, title='style')

    cnn = models.vgg16(pretrained=True).features.to(device).eval()

    content_layers_and_weights = {'conv_4-2': 1.0}
    style_layers_and_weights = {'conv_1-1': 1.0, 'conv_2-1': 0.75, 'conv_3-1': 0.2, 'conv_4-1': 0.2, 'conv_5-1': 0.2}

    model, style_losses, content_losses = get_model_and_losses(cnn=cnn,
                                                               style_img=style_img,
                                                               content_img=content_img,
                                                               style_layers_and_weights=style_layers_and_weights,
                                                               content_layers_and_weights=content_layers_and_weights,
                                                               device=device
                                                               )

    epoch = 2100  # 学習回数：固定
    input_img = content_img.clone()
    # input_img = torch.from_numpy(np.random.uniform(-20, 20, input_img.shape).astype(np.float32)).to('cuda:0')
    optimizer = get_optimizer(input_img)
    # style_weight = 1
    # input_img = style_img : content_weight = 0.05 : style_weight = 100 : conv_4-2 : optimizer : LBFGS : epoch = 150
    content_weight = 1  # コンテンツ画像の重み：固定
    style_weight = 1e5  # スタイル画像の重み：可変  目安：(5e5～5e4)
    total_weight = 1e-1

    run = [0]
    while run[0] <= epoch:

        # 学習ステップ
        def step():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)  # 出力画像をcnnに通す
            style_score = 0
            content_score = 0

            for feature in style_losses:
                style_score += feature.loss

            for feature in content_losses:
                content_score += feature.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = (style_score + content_score) * total_weight
            loss.backward()  # 誤差逆伝播

            run[0] += 1
            if run[0] % 100 == 0:
                print("epoch : ", run[0])
                print('Total Loss : {:4f} Style Loss : {:4f} Content Loss: {:4f}'.format(
                    loss.item(), style_score.item(), content_score.item()))
                print()

            return loss


        optimizer.step(step)
        if run[0] % 300 == 0:  # epoch数300回毎に画像を表示
            image_show(input_img.clone(), title='result')
