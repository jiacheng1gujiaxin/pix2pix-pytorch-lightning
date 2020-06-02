import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import pytorch_lightning as pl

from data.aligned_dataset import AlignedDataset
from . import networks


class Pix2PixModel(pl.LightningModule):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    def __init__(self, input_nc, output_nc, ngf, ndf, netG,
                 gan_mode, norm, n_layers_D, netD, 
                 lr, beta1, device, batch_size,
                 dataroot, load_size, crop_size, preprocess,
                 lambda_L1):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(Pix2PixModel, self).__init__()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G', 'D']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(input_nc, output_nc, ngf, netG, norm)

        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        self.netD = networks.define_D(input_nc + output_nc, ndf, netD,
                                      n_layers_D, norm)

        # define loss functions
        self.criterionCTC = torch.nn.CTCLoss(blank=0)

        self.criterionGAN = networks.GANLoss(gan_mode)
        self.criterionL1 = torch.nn.L1Loss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.lr = lr
        self.beta1 = beta1
        self.batch_size = batch_size
        self.dataroot = dataroot
        self.load_size = load_size
        self.crop_size = crop_size
        self.preprocess = preprocess
        self.lambda_L1 = lambda_L1
        self.a = 1

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A']
        self.real_B = input['B']
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def update_loss_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def update_loss_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
        # third ,G(A) should fake the ocrNet
      
        self.loss_G = self.loss_G_GAN + self.loss_G_L1


        # print(self.loss_G_GAN, self.loss_G_L1/100, self.loss_ocr)
        return self.loss_G


    def training_step(self, batch, batch_nb, optimizer_idx):
        self.set_input(batch)
        self.forward()
        # update D
        if optimizer_idx == 0:
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            d_loss = self.update_loss_D()
            tqdm_dict = {'d_loss': d_loss, 'd_fake': self.loss_D_fake, 'd_real': self.loss_D_real}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            self.logger.log_metrics(tqdm_dict, step=self.global_step)
            return output

        # update G
        if optimizer_idx == 1:
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            g_loss = self.update_loss_G()
            tqdm_dict = {'g_loss': g_loss, 'l1_loss': self.loss_G_L1}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            self.logger.log_metrics(tqdm_dict, step=self.global_step)
            return output

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        return [optimizer_D, optimizer_G], []
    
    def test_step(self, batch, batch_nb):
        print(batch_nb)
        self.set_input(batch)
        fake_B = self.netG(self.real_A)

        self.logger.experiment.add_image('test_real_A', self.real_A[0], self.current_epoch)
        self.logger.experiment.add_image('test_fake_B', fake_B[0], self.current_epoch)
        self.logger.experiment.add_image('test_real_B', self.real_B[0], self.current_epoch)
        
        return 1

    def test_epoch_end(self, outputs):
        print(outputs)
        torch.save(self.netG.state_dict() ,self.logger.log_dir+'/checkpoints'+'/epoch_%s'%self.current_epoch)
    
    
    def validation_step(self, batch, batch_nb):
        self.set_input(batch)
        self.forward()
        d_loss = self.update_loss_D()
        g_loss = self.update_loss_G()
        output_metrics = {
            'g_loss': g_loss, 'l1_loss': self.loss_G_L1,
            'd_loss': d_loss, 'd_fake': self.loss_D_fake, 'd_real': self.loss_D_real,
            'real_A': [],
            'fake_B': [],
            'real_B': [],
        }
        # print(batch_nb, self.a)
        if batch_nb == self.a or batch_nb == 1:
            output_metrics['real_A'] = self.real_A
            output_metrics['fake_B'] = self.fake_B
            output_metrics['real_B'] = self.real_B
        return output_metrics

    def validation_epoch_end(self, outputs):
        self.a = torch.randint(100,(1,))
        g_loss = torch.stack([x['g_loss'] for x in outputs]).mean()
        l1_loss = torch.stack([x['l1_loss'] for x in outputs]).mean()
        d_loss = torch.stack([x['d_loss'] for x in outputs]).mean()
        d_fake = torch.stack([x['d_fake'] for x in outputs]).mean()
        d_real = torch.stack([x['d_real'] for x in outputs]).mean()
        
        # t = [x['real_A'] for x in outputs if x['real_A'] is not None]
        # print(t)
        real_A = [x['real_A'] for x in outputs if x['real_A'] is not None][1][0]
        fake_B = [x['fake_B'] for x in outputs if x['fake_B'] is not None][1][0]
        real_B = [x['real_B'] for x in outputs if x['real_B'] is not None][1][0]

        # self.logger.experiment.add_image('val_real_A', real_A, self.current_epoch)
        # self.logger.experiment.add_image('val_fake_B', fake_B, self.current_epoch)
        # self.logger.experiment.add_image('val_real_B', real_B, self.current_epoch)

        show_img = torch.stack((real_A, real_B, fake_B), dim=0)
        show_img = make_grid(show_img, nrow=3)
        save_image(show_img, self.logger.log_dir+'/epoch_%s.jpg'%self.current_epoch)
        
        tensorboard_logs = {
            'g_loss': g_loss, 'l1_loss': l1_loss,
            'd_loss': d_loss, 'd_fake': d_fake, 'd_real': d_real,
        }

        self.logger.log_metrics(tensorboard_logs, step=self.global_step)

        torch.save(self.netG.state_dict() ,self.logger.log_dir+'/checkpoints'+'/netG_epoch_%s'%self.current_epoch)
        return {'val_loss': g_loss, 'log': tensorboard_logs}
    
    def val_dataloader(self):
        dataset = AlignedDataset(self.dataroot, 'val', self.load_size, self.crop_size, self.preprocess)

        return DataLoader(dataset, batch_size=1, num_workers=4)

    def test_dataloader(self):
        dataset = AlignedDataset(self.dataroot, 'test', self.load_size, self.crop_size, self.preprocess)

        return DataLoader(dataset, batch_size=1, num_workers=4)

    def train_dataloader(self):
        dataset = AlignedDataset(self.dataroot, 'train', self.load_size, self.crop_size, self.preprocess)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)