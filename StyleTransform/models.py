import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms


# 画像読み込み, 前処理
def image_loader(img_path, gray_scale=False, device='cpu', size=512):
    if gray_scale:
        image = Image.open(img_path).convert('L').convert('RGB')  # グレースケールのRGB変換
    else:
        image = Image.open(img_path).convert('RGB')
    loader = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    image = loader(image).unsqueeze(0)  # 次元追加
    image = image[:, [2, 1, 0], :, :]  # 画像をBGR表記に変換
    return image.to(device, torch.float)


# 画像正規化
class Normalization(nn.Module):
    def __init__(self, mean=torch.tensor([0.485, 0.456, 0.406]).to('cuda:0'),
                 std=torch.tensor([0.229, 0.224, 0.225]).to('cuda:0')):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


# コンテンツ損失
class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.weight = weight

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target) * self.weight  # 二乗誤差にレイヤーの重みをかける
        return input


# グラム行列の正規化
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


# スタイル損失
class StyleLoss(nn.Module):

    def __init__(self, target_feature, weight):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.weight = weight

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target) * self.weight
        return input


# 最適化関数
def get_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    # optimizer = optim.Adam([input_img.requires_grad_()], lr=4.0)
    return optimizer


# 一連の動作のモデルと損失を返す
def get_model_and_losses(cnn, style_img, content_img, content_layers_and_weights, style_layers_and_weights, device):
    cnn = copy.deepcopy(cnn)  # ネット―ワークをモデルのcnn層をコピー

    normalization = Normalization().to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization).to(device)  # 学習モデルの生成

    # pooling層を区切りとしてレイヤーに番号をつける
    m = 1
    c = 0
    r = 0
    p = 0
    b = 0
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            c += 1
            name = 'conv_{}-{}'.format(m, c)
        elif isinstance(layer, nn.ReLU):
            r += 1
            name = 'relu_{}-{}'.format(m, r)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            layer = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride)
            m += 1
            p += 1
            c = 0
            r = 0
            b = 0
            name = 'pool_{}-{}'.format(m, p)
        elif isinstance(layer, nn.BatchNorm2d):
            b += 1
            name = 'bn_{}-{}'.format(m, b)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        i += 1

        model.add_module(name, layer)  # cnn層をモデルに追加

        # コンテンツ画像をcnnに通して損失を計算
        if name in content_layers_and_weights.keys():
            target = model(content_img).detach()
            feature_image_show(target, name=name)
            content_loss = ContentLoss(target, content_layers_and_weights[name])
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers_and_weights.keys():
            target_feature = model(style_img).detach()
            feature_image_show(target_feature, name=name)
            style_loss = StyleLoss(target_feature, style_layers_and_weights[name])
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses


# RGB画像の表示
def image_show(tensor, title=None):
    tensor.data.clamp_(0, 1)
    unloader = transforms.ToPILImage()
    tensor = tensor[:, [2, 1, 0], :, :]
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


# 中間層の画像表示
def feature_image_show(images, name):
    images = images.detach().to('cpu').numpy()
    plt.figure(figsize=(20, 20), frameon=False)
    for idx in range(16):
        plt.subplot(4, 4, idx + 1)
        plt.imshow(images[0, idx])
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.title(name)
    plt.show()
