import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from PIL import Image
import os
import glob
from skimage import io
from tqdm import tqdm
import h5py


def statistics_slide(slide_path_list, pos_region_threshold=0.5):
    num_pos_patch_allPosSlide = 0
    num_patch_allPosSlide = 0
    num_neg_patch_allNegSlide = 0
    num_all_slide = len(slide_path_list)

    for i in tqdm(slide_path_list, desc="Statistics"):
        if 'pos' in i.split('/')[-1]:  # pos slide
            # num_pos_patch = len(glob.glob(i + "/*_pos*.jpg"))
            num_pos_patch = 0
            for j in glob.glob(i + "/*_pos*.jpg"):
                pos_ratio = float(j.split("_")[-1].split(".jpg")[0])
                if pos_ratio < pos_region_threshold:
                    continue
                else:
                    num_pos_patch = num_pos_patch + 1
            num_patch = len(glob.glob(i + "/*.jpg"))
            num_pos_patch_allPosSlide = num_pos_patch_allPosSlide + num_pos_patch
            num_patch_allPosSlide = num_patch_allPosSlide + num_patch
        else:  # neg slide
            num_neg_patch = len(glob.glob(i + "/*.jpg"))
            num_neg_patch_allNegSlide = num_neg_patch_allNegSlide + num_neg_patch

    print("[DATA INFO] {} slides totally".format(num_all_slide))
    print("[DATA INFO] pos_patch_ratio in pos slide: {:.4f}({}/{})".format(
        num_pos_patch_allPosSlide / (num_patch_allPosSlide + 1e-5), num_pos_patch_allPosSlide, num_patch_allPosSlide))
    print("[DATA INFO] num of patches: {} ({} from pos slide, {} from neg slide)".format(
        num_patch_allPosSlide+num_neg_patch_allNegSlide, num_patch_allPosSlide, num_neg_patch_allNegSlide))
    return num_patch_allPosSlide + num_neg_patch_allNegSlide


class CAMELYON_16_10x(torch.utils.data.Dataset):
    # @profile
    def __init__(self, root_dir='/home/qlh/Data/CAMELYON16/patches_byDSMIL_224x224_10x',
                 train=True, transform=None, downsample=0.2, pos_region_threshold=0.5, return_bag=False):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.downsample = downsample
        self.pos_region_threshold = pos_region_threshold
        self.return_bag = return_bag
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        if train:
            self.root_dir = os.path.join(self.root_dir, "training")
        else:
            self.root_dir = os.path.join(self.root_dir, "testing")

        all_slides = glob.glob(self.root_dir + "/*")
        # 1.filter the pos slides which have 0 pos patch
        all_pos_slides = glob.glob(self.root_dir + "/*_pos*")

        for i in tqdm(all_pos_slides, desc="Removing Pos Slide without Pos patch"):
            # num_pos_patch = len(glob.glob(i + "/*_pos*.jpg"))
            num_pos_patch = 0
            for j in glob.glob(i + "/*_pos*.jpg"):
                pos_ratio = float(j.split("_")[-1].split(".jpg")[0])
                if pos_ratio < self.pos_region_threshold:
                    continue
                else:
                    num_pos_patch = num_pos_patch + 1
            num_patch = len(glob.glob(i + "/*.jpg"))
            if num_pos_patch/num_patch <= 0.0:  # only remove Pos slide without Pos Patch
                all_slides.remove(i)
                print("[DATA] {} of positive patch ratio {:.4f}({}/{}) is removed".format(
                    i, num_pos_patch/num_patch, num_pos_patch, num_patch))
        # 1.1 down sample the slides
        if self.downsample < 1.0:
            print("================ Down sample ================")
            np.random.shuffle(all_slides)
            all_slides = all_slides[:int(len(all_slides)*self.downsample)]

        statistics_slide(all_slides, self.pos_region_threshold)
        self.num_slides = len(all_slides)
        # 2.extract all available patches and build corresponding labels
        self.all_patches = []
        self.patch_label = []
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        cnt_slide = 0
        cnt_patch = 0
        for i in tqdm(all_slides, ascii=True, desc='preload data'):
            for j in os.listdir(i):
                if "pos" in j:
                    pos_ratio = float(j.split("_")[-1].split(".jpg")[0])
                    if pos_ratio < self.pos_region_threshold:
                        continue
                self.all_patches.append(os.path.join(i, j))
                self.patch_label.append(int('pos' in j))
                self.patch_corresponding_slide_label.append(int('pos' in i.split('/')[-1]))
                self.patch_corresponding_slide_index.append(cnt_slide)
                self.patch_corresponding_slide_name.append(i.split('/')[-1])
                cnt_patch = cnt_patch + 1
            cnt_slide = cnt_slide + 1
        self.num_patches = cnt_patch
        self.all_patches = np.array(self.all_patches)
        self.patch_label = np.array(self.patch_label)
        self.patch_corresponding_slide_label = np.array(self.patch_corresponding_slide_label)
        self.patch_corresponding_slide_index = np.array(self.patch_corresponding_slide_index)
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)

        # 3.do some statistics
        print("[DATA INFO] num_slide is {}; num_patches is {}\npos_patch_ratio is {:.4f}".format(
            self.num_slides, self.num_patches, 1.0*self.patch_label.sum()/self.patch_label.shape[0]))

        print("")

    def __getitem__(self, index):
        if self.return_bag:
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index==index)[0]

            bag = self.all_patches[idx_patch_from_slide_i]
            bag_normed = np.zeros([bag.shape[0], 3, 224, 224], dtype=np.float32)
            for i in range(bag.shape[0]):
                instance_img = io.imread(bag[i])
                bag_normed[i, :, :, :] = self.transform(Image.fromarray(np.uint8(instance_img), 'RGB'))
            bag = bag_normed
            patch_labels = self.patch_label[idx_patch_from_slide_i]
            slide_label = patch_labels.max()
            slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
            slide_name = self.patch_corresponding_slide_name[idx_patch_from_slide_i][0]

            # check data
            if self.patch_corresponding_slide_label[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_label[idx_patch_from_slide_i].min():
                raise
            if self.patch_corresponding_slide_index[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_index[idx_patch_from_slide_i].min():
                raise
            return bag, [patch_labels, slide_label, slide_index, slide_name], index
        else:
            patch_image = io.imread(self.all_patches[index])
            patch_label = self.patch_label[index]
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
            return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        else:
            return self.num_patches


class CAMELYON_16_5x(torch.utils.data.Dataset):
    # @profile
    def __init__(self, root_dir='/home/qlh/Data/CAMELYON16/patches_byDSMIL_224x224_5x',
                 train=True, transform=None, downsample=0.2, pos_region_threshold=0.5, return_bag=False):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.downsample = downsample
        self.pos_region_threshold = pos_region_threshold
        self.return_bag = return_bag
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        if train:
            self.root_dir = os.path.join(self.root_dir, "training")
        else:
            self.root_dir = os.path.join(self.root_dir, "testing")

        all_slides = glob.glob(self.root_dir + "/*")
        # 1.filter the pos slides which have 0 pos patch
        all_pos_slides = glob.glob(self.root_dir + "/*_pos*")

        for i in tqdm(all_pos_slides, desc="Removing Pos Slide without Pos patch"):
            # num_pos_patch = len(glob.glob(i + "/*_pos*.jpg"))
            num_pos_patch = 0
            for j in glob.glob(i + "/*_pos*.jpg"):
                pos_ratio = float(j.split("_")[-1].split(".jpg")[0])
                if pos_ratio < self.pos_region_threshold:
                    continue
                else:
                    num_pos_patch = num_pos_patch + 1
            num_patch = len(glob.glob(i + "/*.jpg"))
            if num_pos_patch/num_patch <= 0.0:  # only remove Pos slide without Pos Patch
                all_slides.remove(i)
                print("[DATA] {} of positive patch ratio {:.4f}({}/{}) is removed".format(
                    i, num_pos_patch/num_patch, num_pos_patch, num_patch))
        # 1.1 down sample the slides
        if self.downsample < 1.0:
            print("================ Down sample ================")
            np.random.shuffle(all_slides)
            all_slides = all_slides[:int(len(all_slides)*self.downsample)]
        statistics_slide(all_slides, self.pos_region_threshold)
        self.num_slides = len(all_slides)
        # 2.extract all available patches and build corresponding labels
        self.all_patches = []
        self.patch_label = []
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        cnt_slide = 0
        cnt_patch = 0
        for i in tqdm(all_slides, ascii=True, desc='preload data'):
            for j in os.listdir(i):
                if "pos" in j:
                    pos_ratio = float(j.split("_")[-1].split(".jpg")[0])
                    if pos_ratio < self.pos_region_threshold:
                        continue
                self.all_patches.append(os.path.join(i, j))
                self.patch_label.append(int('pos' in j))
                self.patch_corresponding_slide_label.append(int('pos' in i.split('/')[-1]))
                self.patch_corresponding_slide_index.append(cnt_slide)
                self.patch_corresponding_slide_name.append(i.split('/')[-1])
                cnt_patch = cnt_patch + 1
            cnt_slide = cnt_slide + 1
        self.num_patches = cnt_patch
        self.all_patches = np.array(self.all_patches)
        self.patch_label = np.array(self.patch_label)
        self.patch_corresponding_slide_label = np.array(self.patch_corresponding_slide_label)
        self.patch_corresponding_slide_index = np.array(self.patch_corresponding_slide_index)
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)

        # 3.do some statistics
        print("[DATA INFO] num_slide is {}; num_patches is {}\npos_patch_ratio is {:.4f}".format(
            self.num_slides, self.num_patches, 1.0*self.patch_label.sum()/self.patch_label.shape[0]))

        # 4. sort patches into bag
        self.all_slides_idx = []
        self.all_slides_label = []
        for i in range(self.patch_corresponding_slide_index.max() + 1):
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index == i)[0]
            bag = self.all_patches[idx_patch_from_slide_i]
            self.all_slides_idx.append(bag)
            patch_labels = self.patch_label[idx_patch_from_slide_i]
            slide_label = patch_labels.max()
            self.all_slides_label.append(slide_label)
        print("")

    def __getitem__(self, index):
        if self.return_bag:
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index==index)[0]

            bag = self.all_patches[idx_patch_from_slide_i]
            bag_normed = np.zeros([bag.shape[0], 3, 224, 224], dtype=np.float32)
            for i in range(bag.shape[0]):
                instance_img = io.imread(bag[i])
                bag_normed[i, :, :, :] = self.transform(Image.fromarray(np.uint8(instance_img), 'RGB'))
            bag = bag_normed
            patch_labels = self.patch_label[idx_patch_from_slide_i]
            slide_label = patch_labels.max()
            slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
            slide_name = self.patch_corresponding_slide_name[idx_patch_from_slide_i][0]

            # check data
            if self.patch_corresponding_slide_label[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_label[idx_patch_from_slide_i].min():
                raise
            if self.patch_corresponding_slide_index[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_index[idx_patch_from_slide_i].min():
                raise
            return bag, [patch_labels, slide_label, slide_index, slide_name], index
        else:
            patch_image = io.imread(self.all_patches[index])
            patch_label = self.patch_label[index]
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
            return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        else:
            return self.num_patches


class CAMELYON_16_5x_feat(torch.utils.data.Dataset):
    # @profile
    def __init__(self, root_dir='/home/qlh/Data/CAMELYON16/patches_byDSMIL_224x224_5x',
                 split='train', return_bag=False, feat="CLIP"):
        self.root_dir = root_dir
        self.split = split
        self.return_bag = return_bag

        # 1. load all featreus and slide label and index
        if feat == 'CLIP' or 'CLIP_RN50':
            save_path = "/home/ubuntu/workspace/MIL_CLIP_New/output_CAMELYON_feat_224x224_5x_CLIP(RN50)"
        elif feat == 'CLIP_ViTB32':
            save_path = "/home/ubuntu/workspace/MIL_CLIP_New/output_CAMELYON_feat_224x224_5x"
        else:
            print("Feature selection not available")
            raise

        if split != 'test':
            self.all_patches = np.array(h5py.File(os.path.join(save_path, "train_patch_feat.h5"), 'r')['dataset_1'])
            self.patch_corresponding_slide_label = np.load(os.path.join(save_path, "train_patch_corresponding_slide_label.npy"))
            self.patch_corresponding_slide_index = np.load(os.path.join(save_path, "train_patch_corresponding_slide_index.npy"))
            self.patch_corresponding_slide_name = np.load(os.path.join(save_path, "train_patch_corresponding_slide_name.npy"))
            self.patch_label = np.load(os.path.join(save_path, "train_patch_label.npy"))
        else:
            self.all_patches = np.array(h5py.File(os.path.join(save_path, "val_patch_feat.h5"), 'r')['dataset_1'])
            self.patch_corresponding_slide_label = np.load(os.path.join(save_path, "val_patch_corresponding_slide_label.npy"))
            self.patch_corresponding_slide_index = np.load(os.path.join(save_path, "val_patch_corresponding_slide_index.npy"))
            self.patch_corresponding_slide_name = np.load(os.path.join(save_path, "val_patch_corresponding_slide_name.npy"))
            self.patch_label = np.load(os.path.join(save_path, "val_patch_label.npy"))

        self.num_patches = self.all_patches.shape[0]
        self.num_slides = self.patch_corresponding_slide_index.max() + 1
        print("[DATA INFO] num_slide is {}; num_patches is {}\npos_patch_ratio is unknown".format(
            self.num_slides, self.num_patches))

        # 2. sort instances features into bag
        self.slide_feat_all = []
        self.slide_label_all = []
        self.slide_patch_label_all = []
        for i in range(self.num_slides):
            idx_from_same_slide = self.patch_corresponding_slide_index == i
            idx_from_same_slide = np.nonzero(idx_from_same_slide)[0]

            self.slide_feat_all.append(self.all_patches[idx_from_same_slide])
            if self.patch_corresponding_slide_label[idx_from_same_slide].max() != self.patch_corresponding_slide_label[
                idx_from_same_slide].min():
                raise
            self.slide_label_all.append(self.patch_corresponding_slide_label[idx_from_same_slide].max())
            self.slide_patch_label_all.append(self.patch_label[idx_from_same_slide])
        print("")

    def __getitem__(self, index):
        if self.return_bag:
            slide_feat = self.slide_feat_all[index]
            slide_label = self.slide_label_all[index]
            slide_patch_label = self.slide_patch_label_all[index]
            return slide_feat, [slide_patch_label, slide_label], index
        else:
            patch_image_feat = self.all_patches[index]
            patch_label = self.patch_label[index]  # [Attention] patch label is unavailable and set to 0
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            return patch_image_feat, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                      patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.num_slides
        else:
            return self.num_patches


def cal_img_mean_std():
    train_ds = CAMELYON_16_5x(train=True, transform=None, downsample=1.0, return_bag=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128,
                                               shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    print("Length of dataset: {}".format(len(train_ds)))
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for data in tqdm(train_loader, desc="Calculating Mean and Std"):
        img = data[0]
        for d in range(3):
            mean[d] += img[:, d, :, :].mean()
            std[d] += img[:, d, :, :].std()
    mean.div_(len(train_ds))
    std.div_(len(train_ds))
    mean = list(mean.numpy()*128)
    std = list(std.numpy()*128)
    print("Mean: {}".format(mean))
    print("Std: {}".format(std))
    return mean, std


if __name__ == '__main__':
    mean, std = cal_img_mean_std()
    transform_data = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.64755785, 0.47759296, 0.657056], std=[0.23896389, 0.26281527, 0.19988984])])  # CAMELYON16_224x224_10x
            # transforms.Normalize(mean=[0.64715815, 0.48541722, 0.65863925], std=[0.24745935, 0.2785922, 0.22133236])])  # CAMELYON16_224x224_5x
    train_ds = CAMELYON_16_10x(train=True, transform=transform_data, downsample=0.01, return_bag=False)
    val_ds = CAMELYON_16_10x(train=False, transform=transform_data, downsample=0.01, return_bag=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32,
                                               shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32,
                                             shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    for data in train_loader:
        patch_img = data[0]
        label_patch = data[1][0]
        label_bag = data[1][1]
        idx = data[-1]
    print("END")
