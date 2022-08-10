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

Over the past twenty years, the Argo program has revolutionized oceanography by deploying thousands of robotic measuring devices known as `floats' across the world. These floats spend most of their time submerged, drifting with the ocean currents. About once a week, a float will change its bouyancy and slowly ascend to surface. During the ascent, measurements of the local pressure, temperatue, salinity and other variables are taken. At the end of the ascent, the float breaches the ocean surface and transmits data to a satellite. Feeling accomplished, the float descends to a depth of around 1km and the process begins again.

<figure>
<img src="/assets/img/argo_bv/float_cycle.png" width="90%">
<figcaption>The Argo float lifecycle. Image from  <a href="https://argo.ucsd.edu/how-do-floats-work/">here</a>.</figcaption>
</figure>

Each cycle produces a one dimensional slice of the ocean known as a profile. Over the course of the Argo program, there have been over two million profiles acquired. Every day, approximately 400 profiles are added to this number. The scale of this data, and the number of scientists who rely upon it, requires standardized means of storage and access. A benefit of this standardization is that data retrieval, processing, and analysis can be automated. This project is a prototype of how this workflow can be turned into an app for interactive data exploration. A lot of this process is already taken care of by the great python library [argopy](https://argopy.readthedocs.io/en/latest/), which is a python wrapper to access data via a REST API.

## Brunt-Väisälä frequency

I decided to calculate the [Brunt-Väisälä (BV) frequency](https://en.wikipedia.org/wiki/Brunt%E2%80%93V%C3%A4is%C3%A4l%C3%A4_frequency) profile from the Argo data and allow the user to interactively see how the profile changes with region and time of year. But what is the BV frequency? In the ocean, density will vary with depth since water is both compressible and its contents (like salinity) change. When ocean currents or an external force disturbs the density versus depth relationship, relatively dense water is shifted upwards and relatively light water is shifted downwards. Gravity then acts as a restoring force, pulling the denser water downwards and the lighter water upwards. The physics is similar to simple harmonic motion, where the density gradient is analogous to the spring constant. Just like harmonic motion, the frequency of oscillation can be determined from the strength of the restoring force. This frequency of oscillation is known as the Brunt-Väisälä (BV) frequency. It is a critical parameter in ocean circulation and is a proxy for the local stability of the water column. BV profiles in the ocean are dependent on a large number of parameters, many of which vary of the course of the year as the mixing from winds, heating from the sun, and local ocean currents change.

## Building The App

<figure>
<img src="/assets/img/argo_bv/data_pipeline.png" width="100%">
<figcaption>Data pipeline for accessing Argo data for Streamlit app.</figcaption>
</figure>