---
layout: page
title: Argo BV 
description: monitoring ocean environments with floats and APIs
img: assets/img/argo_bv/argo_float.jpg
importance: 1
category: Data Science
---

*Making a Streamlit app which interacts with Argo data APIs to retrieve queried data and plot regional and temporal statistics.*

Try the app here! [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/st-howard/ArgoBV/main/ArgoBV.py)

#### Using the App

1. Enter the Latitude and Longitude ranges on the left-hand panel. Click __Update Region__ to draw data on map.
2. Select the profile acquisition date range 
3. Select the depth range of data points to retrieve
4. Select the __Get Data!__ button to retrieve and process data from Argo servers 
5. Adjust the temporal ranges of data to plot, along with plotting options for BV profiles

## Argo: Big Data Oceanography

Over the past twenty years, the Argo program has revolutionized ocean monitoring by deploying thousands of robotic measuring devices known as floats across the world's oceans. At the turn of the last century, a group of international scientists and institutions created the Argo program which aims to monitor and study the world's ocean at scale. 

<figure>
<img src="/assets/img/argo_bv/float_cycle.png" width="90%">
<figcaption>The Argo float lifecycle. Image from  <a href="https://argo.ucsd.edu/how-do-floats-work/">here</a>.</figcaption>
</figure>