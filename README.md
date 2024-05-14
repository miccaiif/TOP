# :sunny: TOP
This is a PyTorch/GPU implementation of the NeurIPS2023 paper [The Rise of AI Language Pathologists: Exploring Two-level Prompt Learning for Few-shot Weakly-supervised Whole Slide Image Classification](https://arxiv.org/abs/2305.17891):

On Updating

The guidance of the data preprocessing is coming soon!

## Frequently Asked Question
* Hi ,what's your data split for few-shot experiments? For example, in one-shot or two-shot setting, how to split training/validation set?

  Hello! Thanks for your attention. The data split for few-shot experiments is controlled by "random seed" directly in the code, e.g. the seed in exp_CAMELYON.sh. We did not manually split the dataset because of the inconvenience. And in specific shots, the experimental settings between different methods are the same. You could benchmark the dataset using the seed by yourself, and maintain that setting to implement various comparison methods. Also, the few-shot WSI classification indeed needs a benchmark dataset, and that is our future direction.

## Citation
If this work is helpful to you, please cite it as:
```
@article{qu2024rise,
  title={The rise of ai language pathologists: Exploring two-level prompt learning for few-shot weakly-supervised whole slide image classification},
  author={Qu, Linhao and Fu, Kexue and Wang, Manning and Song, Zhijian and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

## Acknowledgement
We sincerely thank [CoOp](https://github.com/KaiyangZhou/CoOp) for their inspiration and contributions to the codes.
More information could be referred to the following helpful works:
[DGMIL](https://github.com/miccaiif/DGMIL)
[WENO](https://github.com/miccaiif/WENO)
[DSMIL](https://github.com/binli123/dsmil-wsi)
[CLAM](https://github.com/mahmoodlab/CLAM).

## Contact Information
If you have any question, please email to me [lhqu20@fudan.edu.cn](lhqu20@fudan.edu.cn).
