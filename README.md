# HyperInterval: Hypernetwork approach to training weight interval regions in continual learning

## Abstract
Recently, a new Continual Learning (CL) paradigm was presented to control catastrophic forgetting, called  Interval Continual Learning (InterContiNet), which relies on enforcing interval constraints on the neural network parameter space. 
Unfortunately, InterContiNet training is challenging due to the high dimensionality of the weight space, making intervals difficult to manage. 
To address this issue, we introduce HyperInterval, a technique that employs interval arithmetic within the embedding space and utilizes a hypernetwork to map these intervals to the target network parameter space. We train interval embeddings for consecutive tasks and train a hypernetwork to transform these embeddings into weights of the target network. An embedding for a given task is trained along with the hypernetwork, preserving the response of the target network for the previous task embeddings. Interval arithmetic works with a more manageable, lower-dimensional embedding space rather than directly preparing intervals in a high-dimensional weight space. Our model allows faster and more efficient training. Furthermore, HyperInterval maintains the guarantee of not forgetting. At the end of training, we can choose one universal embedding to produce a single network dedicated to all tasks. In such a framework, hypernetwork is used only for training and can be seen as a meta-trainer.
HyperInterval obtains significantly better results than InterContiNet and gives SOTA results on several benchmarks. 

## Teaser
Train interval embeddings for consecutive tasks and train a hypernetwork to transform these embeddings into weights of the target network.

![Scheme of HyperInterval training method](HyperInterval.png)

## Environment
Use <code>environment.yml</code> file to create a conda environment with necessary libraries: <code>conda env create -f environment.yml</code>.
The [hypnettorch](https://github.com/chrhenning/hypnettorch) package is essential to easily create hypernetworks in [PyTorch](https://pytorch.org/).

## Datasets
For the experiments and ablation study, we use 6 publicly available datasets:
* [Split MNIST](https://arxiv.org/abs/1906.00695)
* [Permuted MNIST-10](https://arxiv.org/abs/1906.00695)
* [Split CIFAR-10](https://arxiv.org/abs/2206.07996) 
* [Split CIFAR-100](https://arxiv.org/abs/2309.14062)
* [TinyImageNet](https://arxiv.org/abs/2309.14062)
* [Permuted MNIST-100](https://arxiv.org/abs/2309.14062) (ablation study)

The datasets may be downloaded when the algorithm runs. For each dataset, the CL task division setup follows the corresponding papers and is specified in our article, supplementary materials.

## Usage

*TO DO*

## Citation

If you use this library in your research project, please cite the following paper: *TO DO*

## License

Copyright 2024 IDEAS NCBR <https://ideas-ncbr.pl/en/> and Group of Machine Learning Research (GMUM), Faculty of Mathematics and Computer Science of Jagiellonian University <https://gmum.net/>.

*[WHICH LICENSE? GNU, Apache 2.0, MIT?]*
