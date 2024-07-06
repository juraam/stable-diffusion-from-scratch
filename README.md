# Stable diffusion from scratch

## Little intro

It is educational repository with purpose to understand how stable diffusion and diffusion models work.

**Why did I decide to create this repo?**<br/>
I started this project with purpose to understand how stable diffusion works and what math it uses. I've read a lot of articles with math explanation and code implementation, but I didn't find something which bridges them in the one place.

**Do you need to have a degree to understand math?**<br/>
No, everyone needs to understand some basic formulas from statistic theory, a little bit algebra, but more important time to spend for analyze and process this information into your brain. Also I try to include links to some basic formulas or good explanation of theory, that you can read if you get stuck.

## How better to read this repo?

Latent diffusion models based on Diffusion models. It's the heart of Stable diffusion and it's really important to understand what diffusion is, how it works and how it is possible to make any picture in our imagination from just a noise. These are my suggestions about steps to understand the information.

1. Start with [Math Explanation.ipynb](https://github.com/juraam/stable-diffusion-from-scratch/blob/main/Math%20Explanation.ipynb) . This is my expanation of math behind diffusion models, based on great articles of other authors and my research. It took two weeks to understand all details of formulas, but I tried to give information easily.

2. In this repo you can look at implementation of two different diffusion models: Conditional and Simple. Conditional is similar to Stable Diffusion, but it works only with input numbers(classes), not text prompts. So it's better for understanding to start from Diffusion models

3. For experiments I created two jupyter notebooks: for training and sampling diffusion and training and sampling conditional diffusion.

## Code

### INSTALL REQUIREMENTS

To start working with code, please download all required dependencies:

```shell
pip install -r requirements.txt
```

### SAMPLING

To get samples you should download ready models. You can download them from github with .

If you want to get random samples, open [Simple diffusion]("Train and sample diffusion.ipynb"). Or if you want to control generation with input of classes, open [Conditional diffusion]("Train and sample conditional diffusion.ipynb").

All jupyter notebooks have PATH_TO_READY_MODEL, which you should fill to skip the training.

### TRAINING

If you want to train a simple diffusion, open [Simple diffusion]("Train and sample diffusion.ipynb"), choose dataset (cifar10 or mnist) and run all cells.

If you want to train a conditional diffusion, open [Conditional diffusion]("Train and sample conditional diffusion.ipynb"), choose dataset (cifar10 or mnist) and run all cells.

## Results of my model

| Model              | FID (CIFAR 10) | FID(MNIST) |
| ------------------ | ------------- | ----------  |
| Original Diffusion | 3.17          | -           |
| My Diffusion       | -             | -           |

## References

* https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
* https://calvinyluo.com/2022/08/26/diffusion-tutorial.html#mjx-eqn%3Aeq%3A79
* https://arxiv.org/abs/2006.11239
* https://arxiv.org/abs/2207.12598
* https://github.com/TeaPearce/Conditional_Diffusion_MNIST
* https://github.com/cloneofsimo/minDiffusion