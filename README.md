![Graffter Banner](images/banner.jpg)
# ELSED: Enhanced Line SEgment Drawing


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iago-suarez/ELSED/blob/main/Python_ELSED.ipynb) [![arXiv](https://img.shields.io/badge/arXiv-2108.03144-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2108.03144)  [![Project Page](https://badgen.net/badge/color/project/green?icon=awesome&label)](https://iago-suarez.com/ELSED)


This repository contains the source code of [**ELSED: Enhanced Line SEgment Drawing**](https://doi.org/10.1016/j.patcog.2022.108619) the fastest line segment detector in the literature. It is ideal for resource-limited devices like drones of smartphones. Visit the [**Project Webpage**](https://iago-suarez.com/ELSED) to try it online!

![Graffter header image](images/header.jpg)

## Dependencies
The code depends on OpenCV (tested with version 4.1.1).
<details> 
<summary>To install OpenCV ... </summary> In Ubuntu 18.04 compile it from sources with the following instructions:

```shell script
# Install dependencies (Ubuntu 18.04)
sudo apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
# Download source code
git clone https://github.com/opencv/opencv.git --branch 4.1.1 --depth 1
# Create build directory
cd opencv && mkdir build && cd build
# Generate makefiles, compile and install
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j
sudo make install
```
</details>

### Using ELSED from python

To install the python wrappers you just have to run:
```
sudo apt-get install libopencv-dev
pip install setuptools
pip install git+https://github.com/iago-suarez/ELSED.git
```

And you can start playing with it:
```python
import pyelsed
import cv2

img = cv2.imread('my_favourite_img.jpg', cv2.IMREAD_GRAYSCALE)
segments, scores = pyelsed.detect(img)
```

### Using ELSED from C++

The code contains a demo detecting large and short line segments in one image.
The code can be compiled with Cmake:

```shell script
mkdir build && cd build
cmake .. && make
./elsed_main
```

The result for the provided image should be:
```
******************************************************
******************* ELSED main demo ******************
******************************************************
ELSED detected: 305 (large) segments
ELSED detected: 391 (short) segments
```

### Cite

```bibtex
@article{suarez2022elsed,
      title={ELSED: Enhanced Line SEgment Drawing}, 
      author={Iago Suárez and José M. Buenaposada and Luis Baumela},
      journal = {Pattern Recognition},
      volume = {127},
      pages = {108619},
      year = {2022},
      issn = {0031-3203},
      doi = {https://doi.org/10.1016/j.patcog.2022.108619},
      url = {https://www.sciencedirect.com/science/article/pii/S0031320322001005}
}
```
