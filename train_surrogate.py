import argparse
import os
import PIL
from PIL import ImageFile
import numpy as np
import torchvision
import torchvision.datasets.folder

from torch.backends import cudnn
from torchvision import transforms as T
import time
from tqdm import tqdm
import kornia.augmentation as Kg
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dct import *


from surrogate import *
from utils import *

PIL.Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True

cudnn.benchmark = False
cudnn.deterministic = True
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser(description='Train Surrogate Model')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int, default=1000)
parser.add_argument('--save_epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10, help='training batch_size')
parser.add_argument('--mode', type=str, default='TSGF')
parser.add_argument('--data_dir', type=str, default='D:/MyPaperCode/Beggin/AGS-master/data/CoCo41K-lmdb/CoCo41K.lmdb')
parser.add_argument('--save_dir', type=str, default='./checkpoints/comics_ffcl001')
parser.add_argument('--inner_step', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--wd', type=float, default=1e-10)
parser.add_argument('--eps_train', type=float, default=1.0)
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
parser.add_argument("--rho", type=float, default=0.5, help="Tuning factor")
parser.add_argument("--sigma", type=float, default=16.0, help="Std of random noise")
parser.add_argument('--gpu', type=str, default='0', help='gpu-id')


args = parser.parse_args()
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def nt_xent_triple_positive(out_1, out_2, out_3, temperature=0.5):
    batch_size = out_1.shape[0]
    out = torch.cat([out_1, out_2, out_3], dim=0)
    # [3*B, 3*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    pos_sim12 = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos_sim13 = torch.exp(torch.sum(out_1 * out_3, dim=-1) / temperature)
    pos_sim23 = torch.exp(torch.sum(out_2 * out_3, dim=-1) / temperature)  # B

    mask = (torch.ones_like(sim_matrix) - torch.eye(3 * batch_size, device=sim_matrix.device)).bool()
    # [3*B, 3*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(3 * batch_size, -1)

    final_loss = (- torch.log(pos_sim12 / (sim_matrix[:batch_size, :].sum(dim=-1) -  pos_sim13)) ).mean()
    final_loss += (- torch.log(pos_sim13 / (sim_matrix[:batch_size, :].sum(dim=-1) -  pos_sim12)) ).mean()
    final_loss += (- torch.log(pos_sim12 / (sim_matrix[batch_size: 2*batch_size, :].sum(dim=-1) -  pos_sim23)) ).mean()
    final_loss += (- torch.log(pos_sim23 / (sim_matrix[batch_size: 2*batch_size, :].sum(dim=-1) -  pos_sim12)) ).mean()
    final_loss += (- torch.log(pos_sim13 / (sim_matrix[2*batch_size:, :].sum(dim=-1) -  pos_sim23)) ).mean()
    final_loss += (- torch.log(pos_sim23 / (sim_matrix[2*batch_size:, :].sum(dim=-1) -  pos_sim13)) ).mean()

    return final_loss / 6.0

def PGD_fs_l2_norand(model, inputs1, inputs2, inputs, eps=1.0, alpha=0.5, iters=1):
    '''
    fs domain L2-norm adversarial attack
    '''
    # Initialize
    model.eval()
    _, out1 = model(inputs1)
    _, out2 = model(inputs2)
    delta = torch.zeros_like(inputs)
    delta = torch.nn.Parameter(delta, requires_grad=True)

    for i in range(iters):
        # Forward pass
        features, out = model(inputs + delta)
        model.zero_grad()

        # Calculate loss
        loss = nt_xent_triple_positive(out1.detach(), out2.detach(), out)
        final_loss = loss

        # Backward pass
        final_loss.backward()

        # Normalize gradients
        grad_norms = torch.norm(delta.grad.view(inputs.size(0), -1), p=2, dim=1) + 1e-10
        grad = delta.grad / grad_norms.view(inputs.size(0), 1, 1, 1)

        # Update delta
        delta.data = delta.data + alpha * grad

        delta.grad = None

        # Clip delta
        delta_norms = torch.norm(delta.data.view(inputs.size(0), -1), p=2, dim=1)
        factor = eps / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))

        delta.data = delta.data * factor.view(-1, 1, 1, 1)
        delta.data = torch.clamp(inputs + delta.data, min=0, max=1) - inputs

    model.train()

    return (inputs + delta).detach()



if __name__ =="__main__":
    '''
    training in lmdb-data format
    '''
    trans_ori = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()])

    trans_f = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        DCT_frequency_enhance()
    ])

    Augmentation_s = nn.Sequential(
        Kg.RandomResizedCrop(size=(224, 224)),
        Kg.RandomHorizontalFlip(p=0.5),
        Kg.ColorJitter(brightness=0.4, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
        Kg.RandomGrayscale(p=0.2),
        Kg.RandomGaussianBlur((int(0.1 * 224 - 1), int(0.1 * 224 - 1)), (0.1, 2.0), p=0.5))

    coff_t = 0.1
    Augmentation_w = nn.Sequential(
        Kg.RandomResizedCrop(size=(224, 224), p=1.0*coff_t),
        Kg.RandomHorizontalFlip(p=0.5*coff_t),
        Kg.ColorJitter(brightness=0.4, contrast=0.8, saturation=0.8, hue=0.2, p=0.8*coff_t),
        Kg.RandomGrayscale(p=0.2*coff_t),
        Kg.RandomGaussianBlur((int(0.1 * 224 - 1), int(0.1 * 224 - 1)), (0.1, 2.0), p=0.5*coff_t))


    os.makedirs(args.save_dir + '/models', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )
    logging.info(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # read from image files.
    dataset = ImageFolderLMDB(db_path=args.data_dir, trans_1=trans_f,trans_2=trans_f,trans_ori=trans_ori)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = Basic_SSL_Model(128).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    for epoch_ind in range(args.start_epoch, args.end_epoch + 1):
        model.train()
        since = time.time()
        tpcl_loss = 0.0
        ail_loss = 0.0
        loss_total = 0.0
        cnt = 0
        print(epoch_ind)
        for iter_ind, (pos_1, pos_2,pos_ori) in tqdm(enumerate(data_loader)):
            pos_1, pos_2, pos_ori = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True),pos_ori.cuda(non_blocking=True)
            inputs_slight_adv = PGD_fs_l2_norand(model, pos_1, pos_2, pos_ori, eps=args.eps_train, alpha=args.eps_train/args.inner_step, iters=args.inner_step)

            inputs_sf_w = Augmentation_w(inputs_slight_adv)
            inputs_sf_s = Augmentation_s(inputs_slight_adv)

            _, out_1 = model(pos_1)
            _, out_2 = model(pos_2)

            _, out_sf_adv_w = model(inputs_sf_w)
            _, out_sf_adv_s = model(inputs_sf_s)

            # final_loss
            loss_f = 1.0 * nt_xent_triple_positive(out_1, out_2, out_sf_adv_w)
            loss_reg = 1 - (F.cosine_similarity(out_sf_adv_w,out_sf_adv_s)).mean()
            loss_final = loss_f + 0.1 * loss_reg

            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()
            tpcl_loss += loss_f.item()
            ail_loss += loss_reg.item()
            loss_total += loss_final.item()

            cnt += 1

            if cnt % 100 == 0:
                print('iter:%d, loss_total:%.2f, tpcl_loss:%.2f,ail_loss:%.2f, time: %ds' % (
                    epoch_ind, loss_total / cnt, tpcl_loss / cnt, ail_loss / cnt,
                    int(time.time() - since)))

        model.eval()
        print('iter:%d, loss_total:%.2f, tpcl_loss:%.2f,ail_loss:%.2f, time: %ds' % (epoch_ind, loss_total/cnt,tpcl_loss/cnt,ail_loss/cnt, int(time.time() - since)))
        logging.info('iter:%d,loss_total:%.2f, tpcl_loss:%.2f, ail_loss:%.2f, time: %ds' % (epoch_ind, loss_total/cnt,tpcl_loss/cnt,ail_loss/cnt, int(time.time() - since)))
        # adjust_learning_rate_linear(args, optimizer, epoch_ind)

        if epoch_ind % args.save_epoch == 0:
            torch.save(model.state_dict(), args.save_dir + f'/models/{args.mode}_{epoch_ind}.pth')



