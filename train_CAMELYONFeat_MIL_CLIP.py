import argparse
import numpy as np
import torch
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
import datetime
import util
import utliz
from tqdm import tqdm
from Datasets_loader.dataset_CAMELYON16_new import CAMELYON_16_5x_feat
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.learnable_prompt import MIL_CLIP, PromptLearner
from copy import deepcopy
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class Map_few_shot(torch.utils.data.Dataset):
    def __init__(self, ds, num_shot=-1):
        self.ds = ds
        self.num_shot = num_shot
        self.few_shot_indexes = []
        # generate few shot idx, compatible with CAMELYON_16_5x_feat
        ds_label = self.ds.slide_label_all
        cate = np.unique(ds_label)
        for cate_i in cate:
            idx_cate_i_all = np.where(ds_label==cate_i)[0]
            if self.num_shot != -1:
                idx_cate_i_few_shot = np.random.choice(idx_cate_i_all, self.num_shot, replace=False).tolist()
            else:
                idx_cate_i_few_shot = idx_cate_i_all.tolist()
            self.few_shot_indexes = self.few_shot_indexes + idx_cate_i_few_shot

        print("{}-shot dataset build".format(num_shot))

    def __getitem__(self, index):
        few_shot_idx = self.few_shot_indexes[index]
        slide_feat, label_list, index_raw = self.ds.__getitem__(few_shot_idx)
        return slide_feat, label_list, index

    def __len__(self):
        return len(self.few_shot_indexes)


class Map_Negative_breaker(torch.utils.data.Dataset):
    def __init__(self, ds, break_p=1.0, break_proportion=0.5):
        self.ds = ds
        self.break_p = break_p
        self.break_proportion = break_proportion

    def __getitem__(self, index):
        slide_feat, label_list, index_raw = self.ds.__getitem__(index)
        if label_list[1] == 0:
            bag_size = slide_feat.shape[0]
            if np.random.rand() < self.break_p:
                idx = np.random.choice(bag_size, int(bag_size*self.break_proportion), replace=False)
                slide_feat = slide_feat[idx]
                label_list[0] = label_list[0][idx]
        return slide_feat, label_list, index

    def __len__(self):
        return len(self.ds)


def get_pathological_tissue_level_prompts(multi_templates=True):
    common_templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a photo of the hard to see {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a close-up photo of a {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a photo of one {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'a low resolution photo of a {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'a photo of the large {}.',
        'a dark photo of a {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
    ]

    pathology_templates = [
        'a histopathological image of {}.',
        'a microscopic image of {} in tissue.',
        'a pathology slide showing {}.',
        'a high magnification image of {}.',
        'an immunohistochemical staining of {}.',
        'a pathology image of {} with inflammatory cells.',
        'a low magnification image of {}.',
        'a pathology image of {} with cellular atypia.',
        'a pathology image of {} with necrosis.',
        'an H&E stained image of {}.',
        'a pathology image of {} with fibrosis.',
        'a pathology image of {} with neoplastic cells.',
        'a pathology image of {} with metastasis.',
        'a pathology image of {} with infiltrating cells.',
        'a pathology image of {} with granulation tissue.',
        'an image of {} on a pathology slide.',
        'a pathology image of {} with edema.',
        'a pathology image of {} with hemorrhage.',
        'a pathology image of {} with degenerative changes.',
        'a pathology image of {} with angiogenesis.',
    ]

    knowledge_from_chatGPT = {
        "Squamous epithelium": "Flat, plate-like cells with a centrally located nucleus.",
        "Columnar epithelium": "Elongated cells with a basally located, oval-shaped nucleus.",
        "Glandular epithelium": "Cells organized in gland-like structures, secreting various substances.",
        "Adipose tissue": "Large, round cells with a thin rim of cytoplasm and a peripheral nucleus, filled with a lipid droplet.",
        "Fibrous connective tissue": "Dense arrangement of collagen fibers and fibroblast cells with elongated nuclei.",
        "Cartilage": "Chondrocytes embedded in a matrix with a basophilic appearance, arranged in lacunae.",
        "Bone tissue": "Calcified matrix with embedded osteocytes in lacunae, connected by canaliculi.",
        "Skeletal muscle": "Long, cylindrical, multinucleated cells with visible striations.",
        "Smooth muscle": "Spindle-shaped cells with a single, centrally located nucleus and no visible striations.",
        "Cardiac muscle": "Branching, striated cells with a single, centrally located nucleus and intercalated discs between cells.",
        "Neurons": "Large, star-shaped cells with a prominent, round nucleus and several processes extending from the cell body.",
        "Glial cells": "Smaller, supportive cells with a less-defined shape and a small, dark nucleus.",
        "Lymphocytes": "Small, round cells with a large, dark nucleus and a thin rim of cytoplasm.",
        "Germinal centers": "Areas of active lymphocyte proliferation and differentiation, appearing as lighter-stained regions in lymphoid tissue.",
        "Erythrocytes": "Anucleate, biconcave, disc-shaped cells.",
        "Leukocytes": "Nucleated white blood cells with various morphologies, including neutrophils, lymphocytes, and monocytes.",
        "Hepatocytes": "Large, polygonal cells with a round, centrally located nucleus and abundant cytoplasm.",
        "Sinusoids": "Vascular channels between hepatocytes, lined by endothelial cells and Kupffer cells in liver tissue.",
        "Glomeruli": "Compact, round structures composed of capillaries and specialized cells with a visible Bowman's space in kidney tissue.",
        "Tubules": "Epithelial-lined structures with various cell types, including proximal and distal tubule cells in kidney tissue.",

        "Carcinoma": "Disorganized tissue architecture, cellular atypia, and possible invasion into surrounding tissues in epithelial-derived tissues.",
        "Sarcoma": "Pleomorphic cells, high cellularity, and possible invasion into surrounding tissues in mesenchymal-derived tissues.",
        "Lymphoma": "Atypical lymphocytes, disrupted lymphoid architecture, and possible effacement of normal lymphoid structures.",
        "Leukemia": "Increased number of abnormal white blood cells in blood smears or bone marrow aspirates, with variable size and nuclear morphology.",
        "Glioma": "Atypical glial cells, increased cellularity, possible necrosis, and disruption of normal central nervous system tissue architecture.",
        "Melanoma": "Atypical melanocytes with variable size, shape, and pigmentation, cellular atypia, and invasion of surrounding tissues."
    }

    knowledge_from_chatGPT_natural = {
        "Squamous epithelium": "Thin, flat cells resembling plates, with a nucleus located in the center.",
        "Columnar epithelium": "Tall cells with an oval-shaped nucleus located toward the base.",
        "Glandular epithelium": "Cells arranged in gland-like structures, responsible for secreting various substances.",
        "Adipose tissue": "Round cells with a thin layer of cytoplasm surrounding a large lipid droplet, and a nucleus pushed to the side.",
        "Fibrous connective tissue": "Tightly packed collagen fibers with elongated nuclei in fibroblast cells.",
        "Cartilage": "Chondrocytes found within a basophilic matrix, situated in small spaces called lacunae.",
        "Bone tissue": "Hard, calcified matrix containing osteocytes in lacunae, which are connected by tiny channels called canaliculi.",
        "Skeletal muscle": "Long, tube-shaped cells with multiple nuclei and visible striations.",
        "Smooth muscle": "Spindle-shaped cells with a single, centrally located nucleus and no visible striations.",
        "Cardiac muscle": "Branched, striated cells with a single central nucleus and intercalated discs connecting the cells.",
        "Neurons": "Star-shaped cells with a large, round nucleus and various extensions coming from the cell body.",
        "Glial cells": "Smaller supporting cells with an undefined shape and a small, dark nucleus.",
        "Lymphocytes": "Tiny, round cells with a large, dark nucleus and a thin layer of cytoplasm.",
        "Erythrocytes": "Disc-shaped cells without a nucleus, featuring a biconcave shape.",
        "Leukocytes": "White blood cells with nuclei, displaying a range of shapes, including neutrophils, lymphocytes, and monocytes.",
        "Hepatocytes": "Sizeable, polygonal cells with a centrally positioned round nucleus and plenty of cytoplasm.",
        "Glomeruli": "Dense, round formations made up of capillaries and special cells, with a visible Bowman's space in kidney tissue.",
        "Tubules": "Structures lined with epithelial cells, containing various cell types like proximal and distal tubule cells in kidney tissue.",
        "Carcinoma": "Cancerous growth originating from epithelial cells, often exhibiting abnormal cell appearance and disordered tissue structure.",
        "Sarcoma": "Cancerous growth arising from mesenchymal cells, such as those found in bone, cartilage, fat, muscle, or blood vessels.",
        "Lymphoma": "Cancerous growth originating from lymphocytes or lymphoid tissue, often marked by unusual lymphocytes and disrupted lymphoid structure.",
        "Leukemia": "Cancerous growth of blood-forming tissues, characterized by a high number of abnormal white blood cells in the blood and bone marrow.",
        "Glioma": "Cancerous growth arising from glial cells in the central nervous system, often displaying abnormal cell appearance, increased cellularity, and tissue decay.",
        "Melanoma": "Cancerous growth originating from melanocytes, often marked by irregular melanocytes, abnormal cell appearance, and invasion into nearby tissues."
    }

    pathology_templates_t = 'an H&E stained image of {}.'
    common_templates_t = 'a photo of the {}.'

    if multi_templates:
        prompts_common_templates = [[common_templates_i.format(condition) for condition in knowledge_from_chatGPT.keys()] for common_templates_i in common_templates]
        prompts_pathology_template = [[pathology_templates_i.format(condition) for condition in knowledge_from_chatGPT.keys()] for pathology_templates_i in pathology_templates]
        prompts_pathology_template_withDescription = [
            [pathology_templates_i.format(tissue_type).replace(".", ", which is {}".format(tissue_description))
             for tissue_type, tissue_description in knowledge_from_chatGPT.items()]
            for pathology_templates_i in pathology_templates]

    else:
        prompts_common_templates = [common_templates_t.format(condition) for condition in knowledge_from_chatGPT.keys()]
        prompts_pathology_template = [pathology_templates_t.format(condition) for condition in knowledge_from_chatGPT.keys()]
        prompts_pathology_template_withDescription = [pathology_templates_t.format(tissue_type).replace(".", ", which is {}".format(tissue_description)) for tissue_type, tissue_description in knowledge_from_chatGPT.items()]

    prompts = [

    ]
    return prompts_common_templates, prompts_pathology_template, prompts_pathology_template_withDescription


class Optimizer:
    def __init__(self, model, train_loader, test_loader, optimizer,
                 writer=None, num_epoch=100,
                 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 weight_lossA=0.0
                 ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer

        self.writer = writer
        self.num_epoch = num_epoch
        self.dev = dev
        self.log_period = 10
        self.weight_lossA = weight_lossA

    def optimize(self):
        for epoch in range(self.num_epoch):
            self.train_one_epoch(epoch)
            if epoch % 10 == 0:
                self.test(epoch)
        return 0

    def train_one_epoch(self, epoch):
        self.model.train()
        loader = self.train_loader

        patch_label_gt = []
        patch_label_pred = []
        patch_label_pred_byMax = []
        bag_label_gt = []
        bag_label_pred = []
        bag_label_pred_byInstance = []
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Epoch {} training'.format(epoch))):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)
            bag_prediction, instance_attn_score = self.model(data.squeeze(0))
            bag_prediction = torch.softmax(bag_prediction, 1)
            loss_D = torch.mean(-1. * (label[1] * torch.log(bag_prediction[:, 1]+1e-5) + (1. - label[1]) * torch.log(1. - bag_prediction[:, 1]+1e-5)))
            instance_attn_score_normed = torch.softmax(instance_attn_score, 0)
            loss_A = torch.triu(instance_attn_score_normed.T @ instance_attn_score_normed, diagonal=1).mean()
            loss = loss_D + self.weight_lossA * loss_A
            if type(self.optimizer) is list:
                for optimizer_i in self.optimizer:
                    optimizer_i.zero_grad()
                loss.backward()
                for optimizer_i in self.optimizer:
                    optimizer_i.step()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if len(instance_attn_score.squeeze().shape) > 1:
                patch_label_pred.append(instance_attn_score.mean(-1, keepdim=True).detach().squeeze())
                patch_label_pred_byMax.append(instance_attn_score.max(-1, keepdim=True)[0].detach().squeeze())
                bag_label_pred_byInstance.append(instance_attn_score.mean(-1, keepdim=True).max().detach().squeeze())
            else:
                patch_label_pred.append(instance_attn_score.detach().squeeze())
                patch_label_pred_byMax.append(instance_attn_score.detach().squeeze())
                bag_label_pred_byInstance.append(instance_attn_score.max().detach().squeeze())
            patch_label_gt.append(label[0].squeeze(0))
            bag_label_pred.append(bag_prediction.mean(0, keepdim=True).detach()[0, 1])
            bag_label_gt.append(label[1])
            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss', loss.item(), niter)
                self.writer.add_scalar('train_loss_A', loss_A.item(), niter)
                self.writer.add_scalar('train_loss_D', loss_D.item(), niter)

        if len(patch_label_pred[0].shape)==0:
            patch_label_pred = torch.Tensor(patch_label_pred)
            patch_label_pred_byMax = torch.Tensor(patch_label_pred_byMax)
        else:
            patch_label_pred = torch.cat(patch_label_pred)
            patch_label_pred_byMax = torch.cat(patch_label_pred_byMax)
        patch_label_gt = torch.cat(patch_label_gt)
        bag_label_pred = torch.tensor(bag_label_pred)
        bag_label_pred_byInstance = torch.stack(bag_label_pred_byInstance)
        bag_label_gt = torch.cat(bag_label_gt)

        self.estimated_AttnScore_norm_para_min = patch_label_pred.min()
        self.estimated_AttnScore_norm_para_max = patch_label_pred.max()
        patch_label_pred_normed = self.norm_AttnScore2Prob(patch_label_pred)
        instance_auc = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred_normed.reshape(-1))

        self.estimated_AttnScore_norm_para_min_byMax = patch_label_pred_byMax.min()
        self.estimated_AttnScore_norm_para_max_byMax = patch_label_pred_byMax.max()
        patch_label_pred_byMax_normed = self.norm_AttnScore2Prob(patch_label_pred_byMax)
        instance_auc_byMax = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred_byMax_normed.reshape(-1))

        bag_auc = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_pred.reshape(-1))
        bag_label_pred_byInstance_normed = self.norm_AttnScore2Prob(bag_label_pred_byInstance)
        bag_auc_byInstance = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_pred_byInstance_normed.reshape(-1))
        self.writer.add_scalar('train_instance_AUC', instance_auc, epoch)
        self.writer.add_scalar('train_instance_AUC_byMax', instance_auc_byMax, epoch)
        self.writer.add_scalar('train_bag_AUC', bag_auc, epoch)
        self.writer.add_scalar('train_bag_AUC_byInstance', bag_auc_byInstance, epoch)

        bag_pred_metrics, _, _ = utliz.cal_TPR_TNR_FPR_FNR(bag_label_gt.reshape(-1), bag_label_pred.reshape(-1))
        self.writer.add_scalar('train_bag_TPR', bag_pred_metrics[0], epoch)
        self.writer.add_scalar('train_bag_TNR', bag_pred_metrics[1], epoch)
        self.writer.add_scalar('train_bag_FPR', bag_pred_metrics[2], epoch)
        self.writer.add_scalar('train_bag_FNR', bag_pred_metrics[3], epoch)

        # patch_pred_metrics, _, _ = utliz.cal_TPR_TNR_FPR_FNR(patch_label_gt.reshape(-1), patch_label_pred_normed.reshape(-1))
        # self.writer.add_scalar('train_patch_TPR', patch_pred_metrics[0], epoch)
        # self.writer.add_scalar('train_patch_TNR', patch_pred_metrics[1], epoch)
        # self.writer.add_scalar('train_patch_FPR', patch_pred_metrics[2], epoch)
        # self.writer.add_scalar('train_patch_FNR', patch_pred_metrics[3], epoch)
        return 0

    def norm_AttnScore2Prob(self, attn_score):
        prob = (attn_score - self.estimated_AttnScore_norm_para_min) / (self.estimated_AttnScore_norm_para_max - self.estimated_AttnScore_norm_para_min)
        return prob

    def norm_AttnScore2Prob_byMax(self, attn_score):
        prob = (attn_score - self.estimated_AttnScore_norm_para_min_byMax) / (self.estimated_AttnScore_norm_para_max_byMax - self.estimated_AttnScore_norm_para_min_byMax)
        return prob

    def test(self, epoch):
        self.model.eval()
        loader = self.test_loader

        patch_label_gt = []
        patch_label_pred = []
        patch_label_pred_byMax = []
        bag_label_gt = []
        bag_label_pred = []
        bag_label_pred_byInstance = []
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Epoch {} testing'.format(epoch))):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)
            with torch.no_grad():
                bag_prediction, instance_attn_score = self.model(data.squeeze(0))
                bag_prediction = torch.softmax(bag_prediction, 1)

            if len(instance_attn_score.squeeze().shape) > 1:
                patch_label_pred.append(instance_attn_score.mean(-1, keepdim=True).detach().squeeze())
                patch_label_pred_byMax.append(instance_attn_score.max(-1, keepdim=True)[0].detach().squeeze())
                bag_label_pred_byInstance.append(instance_attn_score.mean(-1, keepdim=True).max().detach().squeeze())
            else:
                patch_label_pred.append(instance_attn_score.detach().squeeze())
                patch_label_pred_byMax.append(instance_attn_score.detach().squeeze())
                bag_label_pred_byInstance.append(instance_attn_score.max().detach().squeeze())
            patch_label_gt.append(label[0].squeeze(0))
            bag_label_pred.append(bag_prediction.mean(0, keepdim=True).detach()[0, 1])
            bag_label_gt.append(label[1])

        if len(patch_label_pred[0].shape)==0:
            patch_label_pred = torch.Tensor(patch_label_pred)
            patch_label_pred_byMax = torch.Tensor(patch_label_pred_byMax)
        else:
            patch_label_pred = torch.cat(patch_label_pred)
            patch_label_pred_byMax = torch.cat(patch_label_pred_byMax)
        patch_label_gt = torch.cat(patch_label_gt)
        bag_label_prediction = torch.tensor(bag_label_pred)
        bag_label_pred_byInstance = torch.stack(bag_label_pred_byInstance)
        bag_label_gt = torch.cat(bag_label_gt)

        patch_label_pred_normed = (patch_label_pred - patch_label_pred.min()) / (patch_label_pred.max() - patch_label_pred.min())
        instance_auc = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred_normed.reshape(-1))

        patch_label_pred_byMax_normed = (patch_label_pred_byMax - patch_label_pred_byMax.min()) / (patch_label_pred_byMax.max() - patch_label_pred.min())
        instance_auc_byMax = utliz.cal_auc(patch_label_gt.reshape(-1), patch_label_pred_byMax_normed.reshape(-1))

        bag_auc = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_prediction.reshape(-1))
        bag_label_pred_byInstance_normed = self.norm_AttnScore2Prob(bag_label_pred_byInstance)
        bag_auc_byInstance = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_pred_byInstance_normed.reshape(-1))
        self.writer.add_scalar('test_instance_AUC', instance_auc, epoch)
        self.writer.add_scalar('test_instance_AUC_byMax', instance_auc_byMax, epoch)
        self.writer.add_scalar('test_bag_AUC', bag_auc, epoch)
        self.writer.add_scalar('test_bag_AUC_byInstance', bag_auc_byInstance, epoch)

        bag_pred_metrics, _, _ = utliz.cal_TPR_TNR_FPR_FNR(bag_label_gt.reshape(-1), bag_label_prediction.reshape(-1))
        self.writer.add_scalar('test_bag_TPR', bag_pred_metrics[0], epoch)
        self.writer.add_scalar('test_bag_TNR', bag_pred_metrics[1], epoch)
        self.writer.add_scalar('test_bag_FPR', bag_pred_metrics[2], epoch)
        self.writer.add_scalar('test_bag_FNR', bag_pred_metrics[3], epoch)

        # patch_pred_metrics, _, _ = utliz.cal_TPR_TNR_FPR_FNR(patch_label_gt.reshape(-1), patch_label_pred_normed.reshape(-1))
        # self.writer.add_scalar('test_patch_TPR', patch_pred_metrics[0], epoch)
        # self.writer.add_scalar('test_patch_TNR', patch_pred_metrics[1], epoch)
        # self.writer.add_scalar('test_patch_FPR', patch_pred_metrics[2], epoch)
        # self.writer.add_scalar('test_patch_FNR', patch_pred_metrics[3], epoch)
        return 0


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    # optimizer
    parser.add_argument('--epochs', default=500, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr_TB', default=0.001, type=float, help='initial learning rate (default: 0.05) of text branch')
    parser.add_argument('--lr_IB', default=0.001, type=float, help='initial learning rate (default: 0.05) of image branch')
    # parser.add_argument('--lrdrop', default=1000, type=int, help='multiply LR by 0.5 every (default: 150 epochs)')
    # parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: (-5)')
    # parser.add_argument('--dtype', default='f64', choices=['f64', 'f32'], type=str, help='SK-algo dtype (default: f64)')

    # housekeeping
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--workers', default=0, type=int,help='number workers (default: 6)')
    parser.add_argument('--comment', default='Debug_MILCLIP', type=str, help='name for tensorboardX')
    parser.add_argument('--save_intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log_iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    parser.add_argument('--num_shot', default=-1, type=int, help='num of few shot')

    # MIL_CLIP settings
    parser.add_argument('--bagLevel_n_ctx', default=16, type=int, help='num of context')
    parser.add_argument('--instanceLevel_n_ctx', default=16, type=int, help='num of context')
    parser.add_argument('--all_ctx_trainable', default=False, type=str2bool, help='whether all context are trainable')
    parser.add_argument('--csc', default=True, type=str2bool, help='whether use csc')

    parser.add_argument('--pooling_strategy', default='learnablePrompt', type=str,
                        help='pooling strategy in MIL image branch, '
                             'setting to NoCoOp is equivalent to LinearProbe(ABMIL)'
                             'setting to learnablePrompt is equivalent to MIL-CLIP')

    parser.add_argument('--NegBagBreakProb', default=0.0, type=float, help='prob of breaking a negative bag')
    parser.add_argument('--NegBagBreakProP', default=1.0, type=float, help='proportion of breaking a negative bag')
    parser.add_argument('--p_drop_out', default=0.5, type=float, help='prob of drop in instance prompt')
    parser.add_argument('--p_bag_drop_out', default=0.5, type=float, help='prob of drop in instance prompt')

    parser.add_argument('--weight_lossA', default=0.0, type=float, help='weight of LossA')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()

    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+"_%s" % args.comment.replace('/', '_') + \
           "_Seed{}_Bs{}_lrTB{}_lrIB{}_{}Shot_bagLevelNCTX{}_instLevelNCTX{}_AllCTXtrainable{}_CSC{}_poolingStrtegy{}_NegBagProb{}_NegBagProP{}_pDropOut{}_pDropOutBag{}_weightLossA{}".format(
               args.seed, args.batch_size, args.lr_TB, args.lr_IB, args.num_shot,
               args.bagLevel_n_ctx, args.instanceLevel_n_ctx, args.all_ctx_trainable, args.csc,
               args.pooling_strategy,
               args.NegBagBreakProb, args.NegBagBreakProP, args.p_drop_out, args.p_bag_drop_out,
               args.weight_lossA
           )
    try:
        args.device = [int(item) for item in args.device.split(',')]
    except AttributeError:
        args.device = [int(args.device)]
    args.modeldevice = args.device
    util.setup_runtime(seed=args.seed, cuda_dev_id=list(np.unique(args.modeldevice + args.device)))

    print(name, flush=True)

    # Setup loaders
    train_ds_return_bag = CAMELYON_16_5x_feat(split='train', return_bag=True, feat="RN50")
    train_ds_return_bag = Map_few_shot(train_ds_return_bag, num_shot=args.num_shot)
    train_ds_return_bag = Map_Negative_breaker(train_ds_return_bag, break_p=args.NegBagBreakProb, break_proportion=args.NegBagBreakProP)

    val_ds_return_bag = CAMELYON_16_5x_feat(split='test', return_bag=True, feat="RN50")

    train_loader_bag = torch.utils.data.DataLoader(train_ds_return_bag, batch_size=1, shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader_bag = torch.utils.data.DataLoader(val_ds_return_bag, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False)

    # Setup model
    # bagPrompt_ctx_init = ["A normal image patch with regularly shaped cells and smaller, lighter nuclei. * * * * * * * * * *",
    #                       "A tumor image patch with irregular cancerous cells and larger, darker nuclei. * * * * * * * * * *",]
    bagPrompt_ctx_init = ["normal * * * * * * * * * *",
                          "tumor * * * * * * * * * *",]
    bag_prompt_learner = PromptLearner(n_ctx=args.bagLevel_n_ctx,
                                       ctx_init=bagPrompt_ctx_init,
                                       all_ctx_trainable=args.all_ctx_trainable,
                                       csc=args.csc,
                                       classnames=["normal", "tumor"],
                                       clip_model='RN50', p_drop_out=args.p_bag_drop_out)
    prompts_common_templates, prompts_pathology_template, prompts_pathology_template_withDescription = get_pathological_tissue_level_prompts(multi_templates=False)
    instancePrompt_ctx_init = [i + ' * * * * * * * * * *' for i in prompts_pathology_template_withDescription]
    # instancePrompt_ctx_init = ['* * * * * * * * * *' for i in range(1)]
    instance_prompt_learner = PromptLearner(n_ctx=args.instanceLevel_n_ctx,
                                            ctx_init=instancePrompt_ctx_init,
                                            all_ctx_trainable=args.all_ctx_trainable,
                                            csc=args.csc,
                                            classnames=["Prototype {}".format(i) for i in range(len(instancePrompt_ctx_init))],
                                            clip_model='RN50', p_drop_out=args.p_drop_out)
    model = MIL_CLIP(bag_prompt_learner, instance_prompt_learner, clip_model="RN50", pooling_strategy=args.pooling_strategy).to('cuda:0')
    for param in model.text_encoder.parameters():
        param.requires_grad = False

    # Setup optimizer
    # optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer_text_branch = torch.optim.SGD(model.prompt_learner_bagLevel.parameters(), lr=args.lr_TB)
    optimizer_image_branch = torch.optim.SGD(list(model.prompt_learner_instanceLevel.parameters()) +
                                             list(model.pooling.parameters()) +
                                             list(model.coord_trans.parameters()) +
                                             list(model.bag_pred_head.parameters()), lr=args.lr_IB)

    # Setup writer
    writer = SummaryWriter('./runs_CAMELYON_InstP_COOP/%s' % name)
    writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))

    # Start training
    optimizer = Optimizer(model=model, train_loader=train_loader_bag, test_loader=val_loader_bag,
                          optimizer=[optimizer_text_branch, optimizer_image_branch],
                          writer=writer, num_epoch=args.epochs,
                          dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                          weight_lossA=args.weight_lossA)
    optimizer.optimize()

