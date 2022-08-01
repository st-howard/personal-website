---
layout: page
title: Discount DALL-E 
description: exploring text to image methods
img: assets/img/ai_art/cat_dali_ex.png
importance: 1
category: Machine Learning
---

## __Prompt:__
##### *Subject*:  
{% include art_dropdown_context.html %}

##### *Style*:  
{% include art_dropdown_style.html %}

<div class="container">
    <div class="row row-cols-4 no-gutters" id='ai_grid'>
{%- for image in site.static_files -%}
    {%- if image.path contains 'best_examples' -%}
        <div class="col"><img src="{{ site.baseurl }}{{ image.path }}" alt="image" class="img-fluid" /></div>
    {%- endif -%}
{%- endfor -%}
</div>
</div>

<script src="{{site.url}}/assets/js/populate_dropdown.js"></script>
