import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from assessment import assesment_utils
from assessment.assesment_utils import Classifier
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

lidar_cnn = Classifier(1).to(device)
lidar_cnn.load_state_dict(torch.load("assessment/lidar_cnn.pt", weights_only=True))
# Do not unfreeze. Otherwise, it would be difficult to pass the assessment.
for param in lidar_cnn.parameters():
    lidar_cnn.requires_grad = False
lidar_cnn.eval()

IMG_SIZE = 64
img_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),  # Scales data into [0,1]
])


class MyDataset(Dataset):
    def __init__(self, root_dir, start_idx, stop_idx):
        self.classes = ["cubes", "spheres"]
        self.root_dir = root_dir
        self.rgb = []
        self.lidar = []
        self.class_idxs = []

        for class_idx, class_name in enumerate(self.classes):
            for idx in range(start_idx, stop_idx):
                file_number = "{:04d}".format(idx)
                rbg_img = Image.open(self.root_dir + class_name + "/rgb/" + file_number + ".png")
                rbg_img = img_transforms(rbg_img).to(device)
                self.rgb.append(rbg_img)

                lidar_depth = np.load(self.root_dir + class_name + "/lidar/" + file_number + ".npy")
                lidar_depth = torch.from_numpy(lidar_depth[None, :, :]).to(torch.float32).to(device)
                self.lidar.append(lidar_depth)

                self.class_idxs.append(torch.tensor(class_idx, dtype=torch.float32)[None].to(device))

    def __len__(self):
        return len(self.class_idxs)

    def __getitem__(self, idx):
        rbg_img = self.rgb[idx]
        lidar_depth = self.lidar[idx]
        class_idx = self.class_idxs[idx]
        return rbg_img, lidar_depth, class_idx



BATCH_SIZE = 32
VALID_BATCHES = 10
N = 9999

valid_N = VALID_BATCHES*BATCH_SIZE
train_N = N - valid_N

train_data = MyDataset("data/assessment/", 0, train_N)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
valid_data = MyDataset("data/assessment/", train_N, N)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

N *= 2
valid_N *= 2
train_N *= 2

CILP_EMB_SIZE = 200

class Embedder(nn.Module):
    def __init__(self, in_ch, emb_size=CILP_EMB_SIZE):
        super().__init__()
        kernel_size = 3
        stride = 1
        padding = 1

        # Convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(100, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(200, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # Embeddings
        self.dense_emb = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, emb_size)
        )

    def forward(self, x):
        conv = self.conv(x)
        emb = self.dense_emb(conv)
        return F.normalize(emb)



img_embedder = Embedder(4).to(device)
lidar_embedder = Embedder(1).to(device)

class ContrastivePretraining(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_embedder = img_embedder
        self.lidar_embedder = lidar_embedder
        self.cos = nn.CosineSimilarity() #nn.FIXME()

    def forward(self, rgb_imgs, lidar_depths):
        img_emb = self.img_embedder(rgb_imgs)
        lidar_emb = self.lidar_embedder(lidar_depths)

        repeated_img_emb = img_emb.repeat_interleave(len(img_emb), dim=0) #FIXME
        repeated_lidar_emb = lidar_emb.repeat(len(lidar_emb), 1) #FIXME

        similarity = self.cos(repeated_img_emb, repeated_lidar_emb)
        similarity = torch.unflatten(similarity, 0, (BATCH_SIZE, BATCH_SIZE))
        similarity = (similarity + 1) / 2

        logits_per_img = similarity
        logits_per_lidar = similarity.T
        return logits_per_img, logits_per_lidar


def get_CILP_loss(batch):
    rbg_img, lidar_depth, class_idx = batch
    logits_per_img, logits_per_lidar = CILP_model(rbg_img, lidar_depth)
    total_loss = (loss_img(logits_per_img, ground_truth) + loss_lidar(logits_per_lidar, ground_truth))/2 #FIXME, FIXME
    return total_loss, logits_per_img

CILP_model = ContrastivePretraining().to(device)
optimizer = Adam(CILP_model.parameters(), lr=0.0001)
loss_img = nn.CrossEntropyLoss()
loss_lidar = nn.CrossEntropyLoss()
ground_truth = torch.arange(BATCH_SIZE, dtype=torch.long).to(device)
epochs = 3



for epoch in range(epochs):
    CILP_model.train()
    train_loss = 0
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        loss, logits_per_img = get_CILP_loss(batch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    assesment_utils.print_CILP_results(epoch, train_loss/step, logits_per_img, is_train=True)

    CILP_model.eval()
    valid_loss = 0
    for step, batch in enumerate(valid_dataloader):
        loss, logits_per_img = get_CILP_loss(batch)
        valid_loss += loss.item()
    assesment_utils.print_CILP_results(epoch, valid_loss/step, logits_per_img, is_train=False)

for param in CILP_model.parameters():
    CILP_model.requires_grad = False



projector = nn.Sequential(
    nn.Linear(200, 1000),
    ##FIXM
    nn.ReLU(),
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 3200) #FIXM
).to(device)


def get_projector_loss(model, batch):
    rbg_img, lidar_depth, class_idx = batch
    imb_embs = CILP_model.img_embedder(rbg_img) #[32, 200]
    lidar_emb = lidar_cnn.get_embs(lidar_depth)
    pred_lidar_embs = model(imb_embs)
    return nn.MSELoss()(pred_lidar_embs, lidar_emb)


epochs = 40
optimizer = torch.optim.Adam(projector.parameters())
assesment_utils.train_model(projector, optimizer, get_projector_loss, epochs, train_dataloader, valid_dataloader)


##Time to bring it together. Let's create a new model `RGB2LiDARClassifier` where we can use our projector with the pre-trained `lidar_cnn` model.

#**: Please fix the `FIXM`s below. Which `embedder` should we be using from our `CILP_model


class RGB2LiDARClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = projector
        self.img_embedder = CILP_model.img_embedder  #FIXM
        self.shape_classifier = lidar_cnn

    def forward(self, imgs):
        img_encodings = self.img_embedder(imgs)
        proj_lidar_embs = self.projector(img_encodings)
        return self.shape_classifier(data_embs=proj_lidar_embs)



my_classifier = RGB2LiDARClassifier()

def get_correct(output, y):
    zero_tensor = torch.tensor([0]).to(device)
    pred = torch.gt(output, zero_tensor)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct


def get_valid_metrics():
    my_classifier.eval()
    correct = 0
    batch_correct = 0
    for step, batch in enumerate(valid_dataloader):
        rbg_img, _, class_idx = batch
        output = my_classifier(rbg_img)
        loss = nn.BCEWithLogitsLoss()(output, class_idx)
        batch_correct = get_correct(output, class_idx)
        correct += batch_correct
    print(f"Valid Loss: {loss.item():2.4f} | Accuracy {correct/valid_N:2.4f}")

get_valid_metrics()

epochs = 5
optimizer = torch.optim.Adam(my_classifier.parameters())

my_classifier.train()
for epoch in range(epochs):
    correct = 0
    batch_correct = 0
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        rbg_img, _, class_idx = batch
        output = my_classifier(rbg_img)
        loss = nn.BCEWithLogitsLoss()(output, class_idx)
        batch_correct = get_correct(output, class_idx)
        correct += batch_correct
        loss.backward()
        optimizer.step()
    print(f"Train Loss: {loss.item():2.4f} | Accuracy {correct/train_N:2.4f}")
    get_valid_metrics()


