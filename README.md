# MagicClay: Sculpting Meshes with Generative Neural Fields


[Project Page](https://amirbarda.github.io/MagicClay.github.io/) | [Paper](https://arxiv.org/pdf/2403.02460.pdf) |

This is the official implementation of MagicClay. For ease of use, it is implemeneted as a [Threestudio](https://github.com/threestudio-project/threestudio) extension.

![alt text](https://github.com/amir90/MagicClay/blob/main/assets/teaser.png?raw=true)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/O4WQlJ_m_XE/0.jpg)](https://www.youtube.com/watch?v=O4WQlJ_m_XE)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/_qA_5K4QNtg/0.jpg)](https://www.youtube.com/watch?v=_qA_5K4QNtg)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/e6vkxpVnsdI/0.jpg)](https://www.youtube.com/watch?v=e6vkxpVnsdI)


## How to Install and Run


1. follow the installation instruction for [Threestudio](https://github.com/threestudio-project/threestudio)
3. install the following: <br/>
```pip install torch-scatter```<br/>
```pip install cyobj```<br/>
```pip install ConfigArgParse```<br/>
3. put the threestudio-magicclay folder inside the 'custom' folder in the main Threestudio installation.
4. download demo data [here](https://drive.google.com/drive/folders/1FT6CuIwp2qA9JKN2SA6mqg7jabMrbDaf?usp=sharing)
5. put the demo data in a folder called demo_data in the threestudio-magicclay folder, to run the relevant demos in magicclay_demo.sh

If you found this work helpful, please cite as:
```
@article{Barda24,
title = {MagicClay: Sculpting Meshes With Generative Neural Fields},
author = {Amir Barda and Vladimir G. Kim and Noam Aigerman and Amit H. Bermano and Thibault Groueix},
year = {2024},
journal = {SIGGRAPH Asia (Conference track)}}
```

