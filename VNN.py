import torch.nn as nn
import torch
from torchvision import transforms, datasets, models
from torchvision.utils import save_image
from PIL import Image
from matplotlib import colors, pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np

#как импортировать с Dataset все нужные функции????????




def imshow(inp, title=None, plt_ax=plt, default=False):
    """Imshow for tensors"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    plt_ax.grid(False)
    plt.show()

class Generator:
    def __init__(self, model, device, size):
        self.model = model
        self.device = device
        self.size = size

    def prepare(self, image):
        image = image.resize(self.size)
        img = np.array(image)
        img = (img / 255).astype('float32')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = transform(img)
        return img

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def generate(self, link_img1, link_img3):
        img1 = self.load_sample(link_img1)
        img3 = self.load_sample(link_img3)

        img1 = self.prepare(img1)
        img3 = self.prepare(img3)

        prep = torch.cat((img1, img3), 0).unsqueeze(0).to(self.device)
        img2 = self.model(prep)[0].data.cpu()

        return img2

    def safe(self, link_img1, link_img3, filename):
        # save_image(img2, "data/frame" + "10000001.jpg")
        img2= self.generate(link_img1, link_img3)
        img2 = img2.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img2 = std * img2 + mean
        img2 = np.clip(img2, 0, 1)
        im = Image.fromarray((img2 * 255).astype(np.uint8))
        im.save(filename)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1) #Stride = 2
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding= 1)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))

class Encoder(nn.Module):
    def __init__(self, chs=(6, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], 1 if i ==0 else 2) for i in range(len(chs) - 1)])
        #self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
           # x = self.pool(x) #delete last pool
        return ftrs

class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], 1) for i in range(len(chs) - 1)])
        self.final = nn.Conv2d(kernel_size= (1,1),in_channels=self.chs[-1], out_channels=3)

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
           # enc_ftrs = self.crop(encoder_features[i], x)

            enc_ftrs = encoder_features[i]
           # print(list(x.size()),list(enc_ftrs.size()), i)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)


        return self.final(x)

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class UNet(nn.Module):
    def __init__(self, out_sz, enc_chs=(6,64,128,256), dec_chs=(256, 128, 64), num_class=1):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
       # self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.out_sz = out_sz
        #self.final =  nn.Conv2d(kernel_size= (3,3),in_channels= 1, out_channels=3) # padding = 1

    def forward(self, x):

        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
       # out      = self.head(out)
        #out  = self.final(out)


        return out