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

Text to image models have made significant progress recently, with Google's [Imagen](https://imagen.research.google/) ([paper](https://arxiv.org/abs/2205.11487)) and [Parti](https://parti.research.google/) ([paper](https://arxiv.org/abs/2206.10789)), and Open AI's [DALLE-2](https://openai.com/dall-e-2/) ([paper](https://arxiv.org/abs/2204.06125)) achieving start of the art results. Open AI and startups like [Midjourney](https://www.midjourney.com) are creating tiered services for generating images from paying customer prompts. Open source implementations of smaller models with limited compute are available in online web apps, notably [DALLE-mini/craiyon](https://www.craiyon.com/). Top performing models contain billions of parameters, trained on hundreds of millions to billions of images. The complexity, compute, and cost associated with these models suggest that they are beyond the reach of a single individual. However, publicly available weights make results a step below state of the art possible for free on a personal device.  __This is my attempt to run and explore text to image models locally, on my personal desktop computer__. 

## How Text to Image Generation Works

Improvements to text to image generation have been driven by the adaptation of a neural network architecture known as transformers to the domain of image processing and synthesis. [Originally](https://arxiv.org/abs/1706.03762?context=cs) applied to natural language processing (NLP) tasks, transformers learn complex interrelationships between elements in a sequence such as text. Transformers consist of either an encoder, decoder, or both. The encoder transforms data into a fixed length sequence, which roughly corresponds to the algorithm's understanding of what it is seeing in a learned feature space. For the example of translation, the encoder takes a segment of text, say in English, and generates a representation of that text that is language agnostic. The second part of a transformer, a decoder, takes a seqeuence and then learns the mapping of said sequence to an output. In the translation example, it learns how to take the language agnostic representation of a phrase to a different language, such as French. A nice illustrated introduction into transformers can be found [here](https://jalammar.github.io/illustrated-transformer/). 

##### Vision Transformer

A key feature of transformers which makes them difficult to use for image data is self-attention. In self-attention, pairwise relationships between elements of a sequence are learned. Or in other words, one element in the sequence *attends* to another element in same sequence, thus the name self-attention. For images, the number of elements in a sequence describing all pixel values is very large ($$M \times N \times C$$ plus a positional encoding where the image has $$MxN$$ pixels and $$C$$ channels). The [breakthrough](https://arxiv.org/abs/2010.11929) in applying transformers to images was to divide the image into digestablie chunks to create smaller, manageable sequences, as shown below (from [Google's AI blog](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html)).

<p style="text-align:center;">
<img src="https://1.bp.blogspot.com/-_mnVfmzvJWc/X8gMzhZ7SkI/AAAAAAAAG24/8gW2AHEoqUQrBwOqjhYB37A7OOjNyKuNgCLcBGAsYHQ/s1600/image1.gif" width="60%" />
</p>

In the paper introducing vision transformers (ViT), the authors noted that convolutional neural networks actually outperformed transformers when the size of the training dataset was small, and transformers only gained the advantage on very large training datasets. This is due to convolutional nueral networks having a *strong inductive bias*. What does this mean? Convolutional neural networks convolve small matrices/tensors known as kernels to extract important features from an image. While these kernels are learned over time, using these kernels for convolutions tells the algorithm, from an architectural perspective, that local interactions are most important. Conversely, transformers inherently treat all interactions across the image equally to start, with no preference for local features. Therefore, convolution neural nets get a figurative head start on transformers since local features such as edges are very important in images. However, the transformer architecture can eventually learn non-local interactions better than convolutional architectures, given enough training data. Which is why the ViT architecture beat then state of the art algorithms based on recurrent convolutional neural networks, given enough training data.

##### VQGAN

These tradeoffs are described in the introduction of the [*Taming Transformers for Hig-Resolution Image Synthesis*](https://arxiv.org/abs/2012.09841) paper
> "In contrast to the predominant vision architecture, convolutional neural networks (CNNs), the transformer architecture contains no built-in inductive prior on the locality of interactions and is therefore free to learn complex relationships among its inputs. However, this generality also implies that it has to learn all relationships, whereas CNNs have been designed to exploit prior knowledge about strong local correlations within images."

Within this paper, they use the best parts of both convoluational and transformer architectures

> "We hypothesize that low-level image structure is well
described by a local connectivity, i.e. a convolutional architecture, whereas this structural assumption ceases to be effective on higher semantic levels ... Our key insight to obtain a ...model is that, *taken together, convolutional and transformer architectures can model the compositional nature of our visual world*"

and propose a model architecture called *VQGAN*, shown below. At a high level, the VQGAN first encodes local features with a CNN into a latent space $$\hat{z}$$, or a space of learned features. Then a transformer is applied to the latent space, which takes the latent space representation of the image, $$\hat{z}$$, and then learns a quantized $$\mathbf{s}$$ (positionally encoded to $$z_q$$) by predicting the next element of the sequence given the previous elements. This process of predicting the next sequence element from the previous elements is called *autoregressive*. The positionally encoded sequence is then decoded by the CNN decoded and a resulting image is formed. If this process is done in the absence of a prompt, it is called *undconditioned*. If a text prompt is given, the predicition of the next element of the sequence is conditional on the text prompt and the model is *conditioned*.

<p style="text-align:center;">
<img src="https://github.com/CompVis/taming-transformers/raw/master/assets/teaser.png" width="90%" />
</p>

##### Latent Diffusion

An alternative to the autoregressive approach is a process known as *diffusion*. In diffusion, an image is first turned into random noise and then "denoised" by attempting to reverse the steps. A schematic of this processes is shown below

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/06/image-5.png" width="90%">
<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/06/image-4.png" width="90%">

The same [research lab](https://github.com/CompVis) which came up with VQGANs described above, where the transformer is applied to a latent space, applied diffusion to the latent space in the paper [*High-Resolution Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752). The benefit of applying diffusion models to the latent (feature) space is that the high level semantic concepts are efficiently compressed. The model can then learn to create a representation of the image within the latent space from pure noise. Like the VQGAN, the denoising process tries to reproduce a sequence which matches the training data distribution (unconditioned) or matches an expected distribution based on the encoding given to the model (conditioned). Text to image generation is a conditioned version of this process, while the generation of celebrity faces described in the paper is unconditioned.

Fortunately, the latent diffusion model is both computationally light (so I can run it on my personal computer) and has publicly available weights from training on captioned images on the internet. So this is the model I used to create the images at the top of the page. However, latent-diffusion models are more powerful than just text to image generation. The latent diffusion paper also illustrates impressive results for inpainting, semantic image generation, and super resolution.

##### Comparing Top Models

The latest text to image generation paper, [Google's Parti](https://arxiv.org/abs/2206.10789), compares respective models and their architectures

<p style="text-align:center;">
<img src="/assets/img/ai_art/parti_comparisons.png" width="50%" />
</p>

which shows that state of the art performance can be achieved with both autoregressive and diffusion based models. While it is unclear if either the diffusion or autoregressive approach will eventually be superior, the utility of transformers in encoding a compressed, semantic representation of image is apparent.

## Notes on Local Text to Image Generation

- Compare popular prompts
- Stock images in dataset, stock images generated
- Limitations on VRAM
- Complexity in 

## Running Locally
With `conda` installed

#### 1. Clone the latent-diffusion repo
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
