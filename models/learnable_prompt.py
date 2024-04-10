import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import math


_tokenizer = _Tokenizer()


def load_clip_to_cpu(backbone_name="RN50"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model.float()


class AlexNet_CIFAR10_attention(nn.Module):
    def __init__(self, features, num_classes, init=True, withoutAtten=False, input_feat_dim=192*3*3):
        super(AlexNet_CIFAR10_attention, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.withoutAtten=withoutAtten
        self.features = features
        # self.classifier = nn.Sequential(
        #     # nn.BatchNorm1d(1024),
        #     nn.Dropout(0.5),
        #     nn.Linear(input_feat_dim, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True)
        # )
        self.projecter = nn.Sequential(
            nn.Linear(input_feat_dim, 1024),
            nn.BatchNorm1d(1024),
            # nn.ReLU(inplace=True),

            # nn.Dropout(0.5),
            # nn.Linear(input_feat_dim, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(inplace=True)
        )
        self.L = 1024
        self.D = 512
        self.K = 1

        self.attention = nn.Sequential(
            # nn.BatchNorm1d(self.L),
            nn.Linear(self.L, self.D),
            nn.BatchNorm1d(self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.headcount = len(num_classes)
        self.return_features = False
        self.use_projecter = False
        if len(num_classes) == 1:
            self.top_layer = nn.Linear(1024, num_classes[0])
        else:
            for a,i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(4096, i))
            self.top_layer = None  # this way headcount can act as switch.
        if init:
            self._initialize_weights()

    def forward(self, x_raw):
        if self.use_projecter:
            x = self.projecter(x_raw)
        else:
            x = x_raw

        # Attention module
        A_ = self.attention(x)  # NxK
        A_ = torch.transpose(A_, 1, 0)  # KxN
        A = F.softmax(A_, dim=1)  # softmax over N

        x = torch.mm(A, x)  # KxL

        if self.return_features:
            return x, A_

        x = self.top_layer(x)
        return x, 0, A_

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class PromptLearner(nn.Module):
    def __init__(self, n_ctx=0, ctx_init="", all_ctx_trainable=True, csc=True, classnames=["Negative", "Positive"], clip_model='RN50', p_drop_out=0.0):
        # n_ctx: number of context vectors
        # ctx_init: list of strings to initialize context vectors
        # all_ctx_trainable: whether all context vectors are trainable
        # csc: whether to use class-specific context vectors
        # classnames: list of class names
        # clip_model: clip model
        super().__init__()
        self.all_ctx_trainable = all_ctx_trainable
        n_cls = len(classnames)

        clip_model = load_clip_to_cpu(clip_model)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if type(ctx_init) == list:
            self.use_class_specific_ctx = True
        else:
            self.use_class_specific_ctx = False

        if ctx_init:
            if not self.use_class_specific_ctx:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = len(ctx_init.split(" "))
                n_fixed_ctx = len(ctx_init.replace(" *", "").split(" "))
                n_learnable_ctx = ctx_init.count("*")
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                num_nonzero_token = prompt.nonzero().max()
                ctx_vectors = embedding[0, 1: num_nonzero_token , :]
                prompt_prefix = ctx_init
            else:
                prompt_prefix = []
                ctx_vectors = []
                for ctx_init_i in ctx_init:
                    # use given words to initialize context vectors
                    ctx_init_i = ctx_init_i.replace("_", " ")
                    prompt_i = clip.tokenize(ctx_init_i)
                    with torch.no_grad():
                        embedding_i = clip_model.token_embedding(prompt_i).type(dtype)
                    num_nonzero_token = prompt_i.nonzero().max()
                    idx_special_character_i = torch.where(prompt_i == 265)[1]
                    if all_ctx_trainable:
                        ctx_vectors_i = embedding_i[0, 1: num_nonzero_token, :]  # keep ctx between SOS and EOS
                    else:
                        ctx_vectors_i = embedding_i[0, idx_special_character_i, :]
                    prompt_prefix_i = ctx_init_i
                    prompt_prefix.append(prompt_prefix_i)
                    ctx_vectors.append(ctx_vectors_i)

        else:
            # random initialization
            if csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of context words (tokens): {n_ctx}")

        if not self.use_class_specific_ctx:
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        else:
            self.ctx = nn.ParameterList([nn.Parameter(ctx_vector) for ctx_vector in ctx_vectors])

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        if not self.use_class_specific_ctx:
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            prompts = [prompt_prefix_i + " " + name + "." for prompt_prefix_i, name in zip(prompt_prefix, classnames)]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        if not self.use_class_specific_ctx:
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.n_ctx = n_ctx
        else:
            if all_ctx_trainable:
                self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            else:
                for i in range(n_cls):
                    special_character_pos = torch.where(tokenized_prompts[i] == 265)[0].min()  # find first character "*"
                    self.register_buffer("token_prefix_{}".format(i), embedding[i, :special_character_pos, :])
            for i in range(n_cls):
                CLS_pos = torch.where(tokenized_prompts[i] == 265)[0].max()  # find last character "*"
                self.register_buffer("token_suffix_{}".format(i), embedding[i, CLS_pos+1:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        self.drop_layer = torch.nn.Dropout(p=p_drop_out)

    def forward(self):
        if not self.use_class_specific_ctx:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix

            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        else:
            ctx = self.ctx
            prompts = []
            for i in range(self.n_cls):
                if self.all_ctx_trainable:
                    prompt_i = torch.cat(
                        [
                            self.token_prefix[i].unsqueeze(0),  # (n_cls, 1, dim)
                            ctx[i].unsqueeze(0),  # (n_cls, n_ctx, dim)
                            getattr(self, "token_suffix_{}".format(i)).unsqueeze(0)  # (n_cls, *, dim)
                        ],
                        dim=1,
                    )
                else:
                    prompt_i = torch.cat(
                        [
                            getattr(self, "token_prefix_{}".format(i)).unsqueeze(0),  # (n_cls, 1, dim)
                            ctx[i].unsqueeze(0),  # (n_cls, n_ctx, dim)
                            getattr(self, "token_suffix_{}".format(i)).unsqueeze(0)  # (n_cls, *, dim)
                        ],
                        dim=1,
                    )
                prompts.append(prompt_i)
            prompts = torch.cat(prompts, dim=0)
            prompts = self.drop_layer(prompts)
        return prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MIL_CLIP(nn.Module):
    def __init__(self, prompt_learner_bagLevel, prompt_learner_instanceLevel, clip_model="RN50", pooling_strategy='NoCoOp'):
        super().__init__()
        self.prompt_learner_bagLevel = prompt_learner_bagLevel
        self.prompt_learner_instanceLevel = prompt_learner_instanceLevel

        clip_model = load_clip_to_cpu(backbone_name=clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.logit_scale.requires_grad = False
        self.dtype = clip_model.dtype

        self.pooling_strategy = pooling_strategy
        self.pooling = AlexNet_CIFAR10_attention(features=None, num_classes=[2], input_feat_dim=1024)
        if self.pooling_strategy == 'NoCoOp':
            self.pooling.return_features = False  # use CoOp to classify bag feat
            self.pooling.use_projecter = True     # project clip feat to other feat space
        else:
            self.pooling.return_features = True   # use CoOp to classify bag feat
            self.pooling.use_projecter = False    # keep bag feat in raw clip feat space

        self.coord_trans = nn.Sequential(
            nn.Linear(self.prompt_learner_instanceLevel.tokenized_prompts.shape[0], 1),
        )

        # self.projector = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024)
        # )
        #
        self.bag_pred_head = nn.Sequential(
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 2),
        )

    def forward(self, image_raw):

        # the first text token is taken as bag classifier
        bag_prompts = self.prompt_learner_bagLevel()
        bag_text_features = self.text_encoder(bag_prompts, self.prompt_learner_bagLevel.tokenized_prompts)
        bag_text_features = bag_text_features / bag_text_features.norm(dim=-1, keepdim=True)

        # the second text token is taken as instance classifier to give weight for instances pooling
        instance_prompts = self.prompt_learner_instanceLevel()
        instance_text_features = self.text_encoder(instance_prompts, self.prompt_learner_instanceLevel.tokenized_prompts)
        instance_text_features = instance_text_features / instance_text_features.norm(dim=-1, keepdim=True)

        image_raw = image_raw.squeeze()
        # image = self.projector(image_raw)
        image = image_raw

        if self.pooling_strategy == 'NoCoOp':  # special case for experiments, equivalent to baseline linear-probe(ABMIL)
            logits, _, atten_socre = self.pooling(image.squeeze())
            atten_socre = atten_socre.permute(1, 0)
            return logits, atten_socre

        if self.pooling_strategy == 'ABMIL':
            # attention-based pooling
            image_features, attn_score = self.pooling(image.squeeze())
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ bag_text_features.t()
            return logits, attn_score
        elif self.pooling_strategy == 'mean':
            # mean-pooling
            image_features = torch.mean(image.squeeze(), dim=0, keepdim=True)
        elif self.pooling_strategy == 'max':
            # max-pooling
            image_features = torch.max(image.squeeze(), dim=0, keepdim=True)[0]
        elif self.pooling_strategy == 'first-one':
            # max-pooling
            image_features = image.squeeze()[0:1]
        elif self.pooling_strategy == 'CoOp':
            text_features_2 = instance_text_features[1:2]  # only keep the second as classifier
            # logit_scale = self.logit_scale.exp()
            logit_scale = 100
            logits = logit_scale * image @ text_features_2.t()

            weight_from_CoOp = torch.softmax(logits, dim=0)
            image_features = weight_from_CoOp.t() @ image
        elif self.pooling_strategy == 'learnablePrompt':
            # raw_proj_coord = self.logit_scale.exp() * image @ instance_text_features.t()
            raw_proj_coord = image @ instance_text_features.t()
            attn_score = self.coord_trans(raw_proj_coord)
            attn_score_normalized = torch.softmax(attn_score, dim=0)
            image_features = attn_score_normalized.t() @ image

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ bag_text_features.t()

            return logits, attn_score.squeeze()
        elif self.pooling_strategy == 'learnablePrompt_noCoOp':
            # raw_proj_coord = self.logit_scale.exp() * image @ instance_text_features.t()
            raw_proj_coord = image @ instance_text_features.t()
            attn_score = self.coord_trans(raw_proj_coord)
            attn_score_normalized = torch.softmax(attn_score, dim=0)
            image_features = attn_score_normalized.t() @ image

            # logit_scale = self.logit_scale.exp()
            # logits = logit_scale * image_features @ bag_text_features.t()
            logits = self.bag_pred_head(image_features)

            return logits, attn_score.squeeze()
        elif self.pooling_strategy == 'learnablePrompt_argmax':
            # raw_proj_coord = self.logit_scale.exp() * image @ instance_text_features.t()
            raw_proj_coord = image @ instance_text_features.t()
            attn_score = self.coord_trans(raw_proj_coord)
            attn_score_normalized = torch.softmax(attn_score, dim=0)
            image_features = image[attn_score_normalized.argmax(dim=0)]

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ bag_text_features.t()

            return logits, attn_score.squeeze()
        elif self.pooling_strategy == 'learnablePrompt_multi':
            raw_proj_coord = self.logit_scale.exp() * image @ instance_text_features.t()
            # raw_proj_coord = image @ instance_text_features.t()
            attn_score = raw_proj_coord
            attn_score_normalized = torch.softmax(attn_score, dim=0)
            image_features = attn_score_normalized.t() @ image

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ bag_text_features.t()

            return logits, attn_score
        elif self.pooling_strategy == 'learnablePrompt_multi_noCoOp':
            # raw_proj_coord = self.logit_scale.exp() * image @ instance_text_features.t()
            raw_proj_coord = image @ instance_text_features.t()
            attn_score = raw_proj_coord
            attn_score_normalized = torch.softmax(attn_score, dim=0)
            image_features = attn_score_normalized.t() @ image

            # logit_scale = self.logit_scale.exp()
            # logits = logit_scale * image_features @ bag_text_features.t()
            logits = self.bag_pred_head(image_features)

            return logits, attn_score
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ bag_text_features.t()

        return logits, torch.zeros(image_raw.shape[0])


if __name__ == '__main__':
    print("END")

