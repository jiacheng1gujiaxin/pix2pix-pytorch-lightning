from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from models.pix2pix_model import Pix2PixModel

if __name__ == "__main__":

    logger = TensorBoardLogger('tb_logs', name='my_model')

    trainer = Trainer(logger=logger)
    net = Pix2PixModel(input_nc=3, output_nc=3, ngf=64, ndf=64, netG='unet_128',
                 gan_mode='vanilla', norm='instance', n_layers_D=3, netD='n_layers', 
                 lr=0.0002, beta1=0.5, device='cpu', batch_size=3,
                 dataroot='/home/gujiaxin/桌面/AB', load_size=(300, 300),
                crop_size=(300, 300), preprocess=['resize', 'crop', 'flip'], lambda_L1=100.0)

    trainer.fit(net)