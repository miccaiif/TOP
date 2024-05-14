# :sunny: TOP
This is a PyTorch/GPU implementation of the NeurIPS2023 paper [The Rise of AI Language Pathologists: Exploring Two-level Prompt Learning for Few-shot Weakly-supervised Whole Slide Image Classification](https://arxiv.org/abs/2305.17891):

On Updating

The guidance of the data preprocessing is coming soon!

## Frequently Asked Question
* Hi ,what's your data split for few-shot experiments? For example, in one-shot or two-shot setting, how to split training/validation set?

  Hello! Thanks for your attention. The data split for few-shot experiments is controlled by "random seed" directly in the code, e.g. the seed in exp_CAMELYON.sh. We did not manually split the dataset because of the inconvenience. And in specific shots, the experimental settings between different methods are the same. You could benchmark the dataset using the seed by yourself, and maintain that setting to implement various comparison methods. Also, the few-shot WSI classification indeed needs a benchmark dataset, and that is our future direction.
