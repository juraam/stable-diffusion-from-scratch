# Stable diffusion from scratch

## Little intro

Welcome everyone, I would like to start Readme from questions to myself.

**Why did I decide to create this repo?**
I started this project with purpose to understand how stable diffusion works and what math it uses. I've read a lot of articles with math explanation and code implementation, but I didn't find something which bridges them in the one place.

**Do you need to have a degree to understand math?**
No, everyone needs to understand some basic formulas from statistic theory, a little bit algebra, but more important time to spend for analyze and process this information into your brain. Also I try to include links to some basic formulas or good explanation of theory, that you can read if you get stuck.

### How to better read this repo?

Latent diffusion models based on Diffusion models. It's the heart of Stable diffusion and it's really important to understand what diffusion is, how it works and how it is possible to make any picture in our imagination from just a noise. These are my suggestions about steps to understand the information.

1. Start with [Math Explanation.ipynb]("Math Explanation.ipynb") . This is my expanation of math behind diffusion models, based on great articles of other authors and my research. It took two weeks to understand all details of formulas, but I tried to give information easily.

2. In this repo you can look at implementation of two different diffusion models: Conditional and Simple. Conditional is similar to Stable Diffusion, but it works only with input numbers(classes), not text prompts. So it's better for understanding to start from Diffusion models

3. For experiments I created two jupyter notebooks: for training and sampling diffusion and training and sampling conditional diffusion.

## Code

### INSTALL REQUIREMENTS

To start working with code, to download all requirements first:

```bash
pip install -r requirements.txt
```

### SAMPLING

To get samples you should download ready models. You can download them from github with .

If you want to get random samples, open [Simple diffusion]("Train and sample diffusion.ipynb"). Or if you want to control generation with input of classes, open [Conditional diffusion]("Train and sample conditional diffusion.ipynb").

In all jupyter notebooks you should to fill variable PATH_TO_READY_MODEL to skip training.

### TRAINING

If you want to train a simple diffusion, open [Simple diffusion]("Train and sample diffusion.ipynb"), choose dataset (cifar10 or mnist) and run all cells.

If you want to train a conditional diffusion, open [Conditional diffusion]("Train and sample conditional diffusion.ipynb"), choose dataset (cifar10 or mnist) and run all cells.

## Results of my model

MODEL | FID
My
Original Diffusion