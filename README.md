# PV detection model webapp

## Set-up

First, clone the repositories of Kymatio and the WCAM into the folder. 

```
pip install kymatio

cd robust_pv_mapping
git clone https://github.com/gabrielkasmi/spectral-attribution.git
```

Install the required packages:

``` 
pip install -r requirements.txt
```

Finally, download the model weights on this Zenodo repository [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14673918.svg)](https://doi.org/10.5281/zenodo.14673918)
 and make sure that you've downloaded the training dataset BDAPPV, accessible [here](https://zenodo.org/records/12179554).

## Overview

came from https://github.com/gabrielkasmi/robust_pv_mapping


uses model archiver to put on Vertex AI https://github.com/pytorch/serve

uses App engine to host frontend that makes calls to Vertex AI

how to store data?