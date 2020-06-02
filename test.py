import cv2
import os
import torch

from models.networks import define_G
from torchvision.utils import save_image

def process_input(img_path):
    AB = cv2.imread(img_path)
    height, width = AB.shape[:2]
    w2 = int(width / 2)
    A = AB[:, :w2, :]
    B = AB[:, w2:, :]
    h = (height | 31) + 1
    w = (w2 | 31) + 1
    # print(img.shape)
    img = cv2.resize(B, (w, h))
    print(img.shape)
    inp = img.transpose(2, 0, 1)
    inp = (inp/255 - 0.5)/0.5
    inp = torch.from_numpy(inp).unsqueeze(0)
    return inp.float(), img

if __name__ == "__main__":
    model = define_G(input_nc=3, output_nc=3, ngf=64, netG='unet_256', norm='instance')
    model_pth = '/home/zsk/Desktop/guSPACE/pytorch-lightning-pix2pix/tb_logs/my_model/version_25/checkpoints/netG_epoch_25'
    img_dir = '/home/zsk/Desktop/AB/test'
    out_dir = '/home/zsk/Desktop/AB/testout'
    
    model.load_state_dict(torch.load(model_pth, map_location='cuda'))
    imgs = os.listdir(img_dir)
    for img in imgs:
        img_path = os.path.join(img_dir, img)
        inp, image = process_input(img_path)
        res = model(inp)
        save_image(res[0], os.path.join(out_dir, img))