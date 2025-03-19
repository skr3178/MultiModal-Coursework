import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.colors as colors
from matplotlib import animation
import os

from PIL import Image


IMG_SIZE = 64
BATCH_SIZE = 32
VALID_BATCHES = 10
N = 9999


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_func = nn.MSELoss()


def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]


def get_color_gradient(c1, c2, c3, n1, n2):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    c3_rgb = np.array(hex_to_RGB(c3))/255
    mix_pcts_c1_c2 = [x/(n1-1) for x in range(n1)]
    mix_pcts_c2_c3 = [x/(n2-1) for x in range(n2)]
    rgb_c1_c2 = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts_c1_c2]
    rgb_c2_c3 = [((1-mix)*c2_rgb + (mix*c3_rgb)) for mix in mix_pcts_c2_c3]
    rgb_colors = rgb_c1_c2 + rgb_c2_c3
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]


cmap = colors.ListedColormap(get_color_gradient("#000000", "#76b900", "#f1ffd9", 64, 128))


def format_positions(positions):
    return ['{0: .3f}'.format(x) for x in positions]

def get_outputs(model, batch, inputs_idx):
    inputs = batch[inputs_idx].to(device)
    target = batch[-1].to(device)
    outputs = model(inputs)
    return outputs, target

def print_loss(epoch, loss, outputs, target, is_train=True, is_debug=False):
    loss_type = "train loss:" if is_train else "valid loss:"
    print("epoch", str(epoch), loss_type, str(loss))
    if is_debug:
        print("example pred:", format_positions(outputs[0].tolist()))
        print("example real:", format_positions(target[0].tolist()))
        

def train_model(model, optimizer, input_fn, epochs, train_dataloader, valid_dataloader, target_idx=-1):
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            target = batch[target_idx].to(device)
            outputs = model(*input_fn(batch))
            
            loss = loss_func(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / (step + 1)
        train_losses.append(train_loss)
        print_loss(epoch, train_loss, outputs, target, is_train=True)
        
        model.eval()
        valid_loss = 0
        for step, batch in enumerate(valid_dataloader):
            target = batch[target_idx].to(device)
            outputs = model(*input_fn(batch))
            valid_loss += loss_func(outputs, target).item()
        valid_loss = valid_loss / (step + 1)
        valid_losses.append(valid_loss)
        print_loss(epoch, valid_loss, outputs, target, is_train=False)
    return train_losses, valid_losses


img_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),  # Scales data into [0,1]
])

def get_torch_xyza(lidar_depth, azimuth, zenith):
    x = lidar_depth * torch.sin(-azimuth[:, None])
    z = lidar_depth * torch.sin(-zenith[None, :])
    y = lidar_depth * torch.cos(-azimuth[:, None]) * torch.cos(-zenith[None, :])
    a = torch.where(lidar_depth < 50.0, torch.ones_like(lidar_depth), torch.zeros_like(lidar_depth))
    xyza = torch.stack((x, y, z, a))
    return xyza


class ReplicatorDataset(Dataset):
    def __init__(self, root_dir, start_idx, stop_idx):
        self.root_dir = root_dir
        self.rgb_imgs = []
        self.lidar_depths = []
        self.positions = np.genfromtxt(
            root_dir + "positions.csv", delimiter=",", skip_header=1
        )[start_idx:stop_idx]

        azimuth = np.load(self.root_dir + "azimuth.npy")
        zenith = np.load(self.root_dir + "zenith.npy")
        self.azimuth = torch.from_numpy(azimuth).to(device)
        self.zenith = torch.from_numpy(zenith).to(device)

        for idx in range(start_idx, stop_idx):
            file_number = "{:04d}".format(idx)
            rbg_img = Image.open(self.root_dir + "rgb/" + file_number + ".png")
            rbg_img = img_transforms(rbg_img).to(device)
            self.rgb_imgs.append(rbg_img)

            lidar_depth = np.load(self.root_dir + "lidar/" + file_number + ".npy")
            lidar_depth = torch.from_numpy(lidar_depth).to(torch.float32).to(device)
            self.lidar_depths.append(lidar_depth)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        rbg_img = self.rgb_imgs[idx]
        lidar_depth = self.lidar_depths[idx]
        lidar_xyza = get_torch_xyza(lidar_depth, self.azimuth, self.zenith)

        position = self.positions[idx]
        position = torch.from_numpy(position).to(torch.float32).to(device)

        return rbg_img, lidar_xyza, position
    

def get_replicator_dataloaders(root_dir):
    train_data = ReplicatorDataset(root_dir, 0, N-VALID_BATCHES*BATCH_SIZE)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_data = ReplicatorDataset(root_dir, N-VALID_BATCHES*BATCH_SIZE, N)
    valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    return train_data, train_dataloader, valid_data, valid_dataloader

def animate_3D_points(file_path, fig, ax, x, y, z, c):
    def anim_init():
        ax.scatter(x, y, z, color=c, marker='o')
        return fig,

    def animate(i):
        ax.view_init(elev=30., azim=i)
        return fig,
    
    if not os.path.exists(file_path):
        anim = animation.FuncAnimation(
            fig, animate, init_func=anim_init, frames=360, interval=20, blit=True
        )
        anim.save(file_path, fps=30)