# GLANN
Official code for paper "[Non-Adversarial Image Synthesis with Generative Latent Nearest Neighbors](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hoshen_Non-Adversarial_Image_Synthesis_With_Generative_Latent_Nearest_Neighbors_CVPR_2019_paper.pdf)" by Yedid Hoshen, Ke Li and Jitendra Malik, CVPR'19

This repository contains Python3 implementations of:
- Generative Latent Optimization (GLO)
- Implicit Maximum Likelihood Estimation (IMLE)
- Generative Latent Nearest Neighbors (GLANN)

# Quick Start

Install dependencies:
```bash
pip install numpy scipy pytorch torchvision python-mnist fbpca faiss
```

Edit prepare_mnist.py with the correct path to the data.

Prepare a dataset:
```bash
python prepare_mnist.py
```

Train GLO on a particular config:
```bash
python train_glo.py configs/mnist.yaml
```

Train IMLE based on the trained GLO model:
```bash
python train_icp.py configs/mnist.yaml
```

Evaluate the FID of the trained GLANN model:
```bash
python test_fid.py configs/mnist.yaml
```

## References
Please cite [[1]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hoshen_Non-Adversarial_Image_Synthesis_With_Generative_Latent_Nearest_Neighbors_CVPR_2019_paper.pdf) if you found the resources in this repository useful.

### Non-Adversarial Image Synthesis with Generative Latent Nearest Neighbors

[1] Y. Hoshen, K. Li, J. Malik, [*Non-Adversarial Image Synthesis with Generative Latent Nearest Neighbors*](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hoshen_Non-Adversarial_Image_Synthesis_With_Generative_Latent_Nearest_Neighbors_CVPR_2019_paper.pdf)
```
@inproceedings{hoshen2019non,
  title={Non-Adversarial Image Synthesis with Generative Latent Nearest Neighbors},
  author={Hoshen, Yedid and Li, Ke and Malik, Jitendra},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5811--5819},
  year={2019}
}
```

### Related work
* [P. Bojanowski, A. Joulin, D. Lopez-Paz, A. Szlam - Optimizing the Latent Space of Generative Networks, 2017](https://arxiv.org/abs/1707.05776)
* [K. Li, J. Malik - Implicit Maximum Likelihood Estimation, 2018](https://arxiv.org/abs/1809.09087)
