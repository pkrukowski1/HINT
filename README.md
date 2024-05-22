# HyperInterval: Hypernetwork approach to training weight interval regions in continual learning

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
