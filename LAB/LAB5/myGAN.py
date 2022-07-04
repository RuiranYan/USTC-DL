# import lib
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# globle para
dir = './data/'
noise_size = 100
gen_fm = 32  # generator feature map
dis_fm = 32  # discriminator feature map
batch_size = 256
d_every = 1  # train discriminator every batch
g_every = 5  # train generator every 5 batch
n_epoch = 200


class G(nn.Module): # Generator
    def __init__(self, n_input, n_fm):  # n_input:noiseSize n_fm:feature map
        super(G, self).__init__()
        self.generator = nn.Sequential(
            # ConvTranspose2d: h -> (h-1) * stride - 2 * padding + Size
            nn.ConvTranspose2d(n_input, 8 * n_fm, 4, 1, 0),  # 1 -> (1-1)*1+1*(4-1)+0+1=4
            nn.BatchNorm2d(8 * n_fm),
            nn.ReLU(True),  # n_input,1,1 -> 8*n_fm,4,4

            nn.ConvTranspose2d(8 * n_fm, 4 * n_fm, 4, 2, 1),  # 4 -> (4-1)*2-2*1+4=8
            nn.BatchNorm2d(4 * n_fm),
            nn.ReLU(True),  # 8*n_fm,4,4 -> 4*n_fm,8,8

            nn.ConvTranspose2d(4 * n_fm, 2 * n_fm, 4, 2, 1),  # 8 -> (8-1)*2-2*1+4=16
            nn.BatchNorm2d(2 * n_fm),
            nn.ReLU(True),  # 4*n_fm,8,8 -> 2*n_fm,16,16

            nn.ConvTranspose2d(2 * n_fm, n_fm, 4, 2, 1),  # 16 -> (16-1)*2-2*1+4=32
            nn.BatchNorm2d(n_fm),
            nn.ReLU(True),  # 2*n_fm,16,16 -> n_fm,32,32

            nn.ConvTranspose2d(n_fm, 3, 5, 3, 1),  # 32 -> (32-1)*3-2*1+5=96    n_fm,32,32 -> 3,96,96
            nn.Tanh()

        )

    def forward(self, x):
        return self.generator(x)


class D(nn.Module): # Discriminator
    def __init__(self, n_fm):
        super(D, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, n_fm, 5, 3, 1),
            nn.LeakyReLU(0.2, inplace=True),  # n_fm * 32 * 32

            nn.Conv2d(n_fm, 2 * n_fm, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),  # 2*n_fm * 16 * 16

            nn.Conv2d(2 * n_fm, 4 * n_fm, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),  # 4*n_fm * 8 * 8

            nn.Conv2d(4 * n_fm, 8 * n_fm, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),  # 8*n_fm * 4 * 4

            nn.Conv2d(8 * n_fm, 1, 4, 1, 0),  # 1*1*1
            nn.Sigmoid()

        )

    def forward(self, input):
        return self.discriminator(input).view(-1)


def TrainAndTest(dataloader, optimizer_d, optimizer_g, Generator, Discriminator, criterion):
    true_labels = Variable(torch.ones(batch_size))
    fake_labels = Variable(torch.zeros(batch_size))
    noises = Variable(torch.randn(batch_size, noise_size, 1, 1))
    gen_noises = torch.randn(batch_size, noise_size, 1, 1) # fix noise to observe the developments


    if torch.cuda.is_available() == True:
        Generator.cuda()
        Discriminator.cuda()
        criterion.cuda()
        true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()
        noises = noises.cuda()
        gen_noises = gen_noises.cuda()

    loss_d_list = []
    loss_g_list = []

    for epoch in range(1, n_epoch + 1):
        print("epcoh:{}".format(epoch))

        loss_d = 0.0
        loss_g = 0.0
        n_d = 0  # discriminator train num
        n_g = 0  # generator train num
        for i, (image, _) in enumerate(tqdm(dataloader)):
            real_image = Variable(image)
            if torch.cuda.is_available() == True:
                real_image = real_image.cuda()

            if (i + 1) % d_every == 0:
                optimizer_d.zero_grad()
                output = Discriminator(real_image)
                error_d_real = criterion(output, true_labels)
                # print("epcoh:{}, discriminator real error:{}".format(epoch, error_d_real))
                error_d_real.backward()
                loss_d += error_d_real.item()
                n_d += 1

                noises.data.copy_(torch.randn(batch_size, noise_size, 1, 1)) # new noise
                fake_img = Generator(noises).detach()  # detach to avoid backward
                fake_output = Discriminator(fake_img)
                error_d_fake = criterion(fake_output, fake_labels)
                # print("epcoh:{}, discriminator fake error:{}".format(epoch,error_d_fake))
                error_d_fake.backward()
                loss_d += error_d_fake.item()
                n_d += 1
                optimizer_d.step()

            if (i + 1) % g_every == 0:
                optimizer_g.zero_grad()
                noises.data.copy_(torch.randn(batch_size, noise_size, 1, 1))
                fake_img = Generator(noises)  # no detach for training generator
                fake_output = Discriminator(fake_img)
                error_g = criterion(fake_output, true_labels)
                error_g.backward()
                loss_g += error_g.item()
                n_g += 1
                optimizer_g.step()

        loss_d = loss_d / n_d
        loss_g = loss_g / n_g
        print("epcoh:{}, loss_d:{}, loss_g:{}".format(epoch, loss_d, loss_g))
        loss_d_list.append(loss_d)
        loss_g_list.append(loss_g)

        # test
        if epoch % 10 == 0 or epoch == 1:

            fake_imags = Generator(gen_noises)
            fake_imags = fake_imags.data.cpu()[:64] * 0.5 + 0.5  # back norm

            # plt
            fig = plt.figure()
            for i, image in enumerate(fake_imags):
                ax = fig.add_subplot(8, 8, i + 1)
                plt.axis('off')
                plt.imshow(image.permute(1, 2, 0))
            plt.suptitle('epoch=%d' % epoch)
            # plt.show()
            plt.savefig('./img/epoch-{}.png'.format(epoch))
            print("epoch %d result saved" % epoch)

            plt.figure()
            plt.title("error curve")
            plt.xlabel("epoch")
            plt.ylabel("error")
            plt.plot(loss_d_list, label="discriminator loss")
            plt.plot(loss_g_list, label="generator loss")
            plt.legend()

            plt.savefig('./img/epoch-{}-loss.png'.format(epoch))
            print("epoch %d loss saved" % epoch)


if __name__ == '__main__':
    print("data loading...")
    transform = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # (0,1) -> (-1,1)
    ])

    dataset = torchvision.datasets.ImageFolder(dir, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                             drop_last=True)
    print('data loading done')
    Generator = G(noise_size, gen_fm)
    Discriminator = D(dis_fm)

    optimizer_g = torch.optim.Adam(Generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(Discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = torch.nn.BCELoss()

    TrainAndTest(dataloader, optimizer_d, optimizer_g, Generator, Discriminator, criterion)

