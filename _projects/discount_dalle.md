---
layout: page
title: Discount DALL-E 
description: exploring text to image methods
img: assets/img/ai_art/cat_dali_ex.png
importance: 1
category: Machine Learning
---

*Images below are generated with code from the [latent-diffusion](https://github.com/CompVis/latent-diffusion) repository, weights pre-trained on the [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) dataset*

## __Prompt:__
##### *Subject*:  
{% include art_dropdown_context.html %}

##### *Style*:  
{% include art_dropdown_style.html %}

<div class="container">
<div class="row row-cols-4 no-gutters" id='ai_grid'>
</div>
</div>
<br>

## Intro

Text to image models have made significant progress recently, with Google's [Imagen](https://imagen.research.google/) ([paper](https://arxiv.org/abs/2205.11487)) and [Parti](https://parti.research.google/) ([paper](https://arxiv.org/abs/2206.10789)), and Open AI's [DALLE-2](https://openai.com/dall-e-2/) ([paper](https://arxiv.org/abs/2204.06125)) achieving start of the art results. Open AI and startups like [Midjourney](https://www.midjourney.com) are creating tiered services for generating images from paying customer prompts. Open source implementations of smaller models with limited computed are available in online web apps, notably [DALLE-mini/craiyon](https://www.craiyon.com/). Top performing models contain billions of parameters, trained on hundreds of millions to billions of images. The complexity, compute, and cost associated with these models suggest that they are beyond the reach of a single individual. However, publicly available weights make results a step below state of the art possible for free on a personal device.  __This is my attempt to run and explore text to image models locally, on my personal desktop computer__. 

## How Text to Image Generation Works



## Running Locally
With `conda` installed

#### 1. Clone the latent-diffusion rep
```
git clone https://github.com/CompVis/latent-diffusion.git
```

#### 2. Create the conda virtual environment
```
cd latent-diffusion
conda env create -f environment.yaml
conda activate ldm
```
Note: Need to install the appropriate version of pytorch. If you have an Nvidia GPU, the CUDA version. Otherwise, the CPU version is needed. If using the CPU version, then the `txt2img.py` script will need to be adjusted since it contains code specifically for CUDA. I was lucky to have an Nvidia GPU, so the code worked out of the box. Info for downloading a specific configuration of Pytorch can be found [here](https://pytorch.org/get-started/locally/).

#### 3. Download the pretrained weights
```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

#### 4. Run the script!
```
python scripts/txt2img.py --prompt "Your prompt here!"
```
The image outputs will be default 256x256 and appear in the outputs folder local to the repo. Look into the `txt2img.py` script to see the adjustable parameters. For the above images, I set the diffusion dimensions to 250, which was the level of diminishing returns mentioned on the latent-diffusion README. I also used the `--scale` set to 7.5. The scale is related to "Classifier-free guidance", which is a relative weighting to conditioned image generation (conditioned in this case is the text prompt) and unconditioned image generation in the sampling process. 

#### 5. (Optional) Upsample the image using ERSGAN



<script src="{{site.url}}/assets/js/populate_dropdown.js"></script>
