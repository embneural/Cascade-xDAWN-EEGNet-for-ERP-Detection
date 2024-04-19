<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III">
    <img src="logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">SOTA of P300 Detection in BCI competition II, III</h3>
<!-- PROJECT LOGO -->
  <p align="center">
    <a href="https://github.com/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III">
    <strong>
    </strong>
    <a href="https://github.com/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III">View Demo</a>
    ·
    <a href="https://github.com/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III/issues">Report Bug</a>
    ·
    <a href="https://github.com/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III/issues">Request Feature</a>
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS 
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
-->

<!-- 
[![Product Name Screen Shot][product-screenshot]](https://example.com)
Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `wzhcoder`, `SOTA-of-P300-Detection-in-BCI-Competitions-II-III`, `twitter_handle`, `linkedin_username`, `3516766936@qq.com_client`, `3516766936@qq.com`, `SOTA of P300 Detection in BCI competition II, III`, `project_description`
<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->

<!-- 
### Built With
* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]
<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->

<!-- ABOUT THE PROJECT -->
## About The Project
This project aims to provide a simple general framework to help beginners who are struggling to achieve the state of the art of p300 detection in [BCI competitions II and III](https://www.bbci.de/competition/). Two of the most popular deep learning models( i.e., EEGNet, DeepConvNet) and [Improved EEGNet](https://journals.sagepub.com/doi/full/10.26599/BSA.2022.9050007) (which won second place on the within-subject RSVP paradigm in WRC 2021) are adopted to serve as the base models for this project.

<!-- GETTING STARTED -->
## Getting Started
### Prepare Datasets
1. Download [Data set IIb: ‹P300 speller paradigm›](https://www.bbci.de/competition/ii/) and [Data set II: ‹P300 speller paradigm›](https://www.bbci.de/competition/iii/). Dataset IIb only contains one subject, while dataset II has two subjects (A and B).
2. Download the true label of tests sets of [dataset IIa](https://www.bbci.de/competition/ii/results/labels_data_set_iib.txt) and the true label of tests sets of dataset II ( [A](https://www.bbci.de/competition/iii/results/albany/true_labels_a.txt) and [B](https://www.bbci.de/competition/iii/results/albany/true_labels_b.txt)). I have downloaded those labels of test sets and put them in the [folder](dataset_labels).
3. Put those true labels into the root directory of the downloaded dataset, respectively.

### Requriments
- pytorch
- mne
- pyriemann
- sklearn
- visdom


<!-- USAGE EXAMPLES -->
### Usage
1. Change src_path and root in main.py
2. Run visdom in the console.
```sh
   visdom
```
3. Run main.py in the console.
```sh
   python main.py
```


## Methods
1. **ESVMs**. link: [P300 Detection using Ensemble of SVM for
Brain-Computer Interface Application](http://dspace.nitrkl.ac.in:8080/dspace/bitstream/2080/3022/1/2018_ICCNT_SKundu_P300.pdf).
2. **CNN-1** and **MCNN-1**.  link: [Convolutional neural networks for P300 detection with application to brain-computer interfaces](https://liacs.leidenuniv.nl/~stefanovtp/pdf/IJCAI_18.pdf).
3. **OLCNN**.  link: [A Simple Convolutional Neural Network
for Accurate P300 Detection and Character Spelling
in Brain Computer Interface](https://www.ijcai.org/Proceedings/2018/0222.pdf).
4. **BN3**. link: [Deep learning based on batch normalization for P300 signal detection](https://www.sciencedirect.com/science/article/abs/pii/S0925231217314601).
5. **ERP-CaspNet**.  link: [Capsule network for ERP detection in brain-computer interface](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9393395).
6. **MsCNN-TL-ESVM**.  link: [MsCNN: A Deep Learning Framework for P300
Based Brain-Computer Interface Speller](https://www.researchgate.net/profile/Sourav-Kundu-4/publication/345380593_MsCNN_A_Deep_Learning_Framework_for_P300-Based_Brain-Computer_Interface_Speller/links/60f955f1169a1a0103ab8381/MsCNN-A-Deep-Learning-Framework-for-P300-Based-Brain-Computer-Interface-Speller.pdf).
7. **ST-CaspNet**.  link: [ST-CapsNet: Linking Spatial and Temporal Attention with Capsule Network for P300 Detection Improvement](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10018278).
8. **EEGNet**.  link: [EEGNet: A Compact Convolutional Neural Network
for EEG-based Brain-Computer Interfaces](https://arxiv.org/pdf/1611.08024.pdf).
9. **Improved EEGNet**.  link:[An improved EEGNet for single-trial EEG classification in rapid serial visual presentation task
](https://journals.sagepub.com/doi/full/10.26599/BSA.2022.9050007).
10. **DeepConvNet**.  link: [Deep Learning With Convolutional Neural
Networks for EEG Decoding and Visualization](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/hbm.23730).
11. **DeepConvNet and EEGNet are good enough**. link: [A review of deep learning methods for cross‐subject rapid serial
visual presentation detection in World Robot Contest 2022](https://www.researchgate.net/profile/Hongtao-Wang-16/publication/371782540_A_Review_of_Deep_Learning_Methods_for_Cross-subject_Rapid_Serial_Visual_Presentation_Detection_in_World_Robot_Contest_2022/links/649e47af8de7ed28ba64b37c/A-Review-of-Deep-Learning-Methods-for-Cross-subject-Rapid-Serial-Visual-Presentation-Detection-in-World-Robot-Contest-2022.pdf)

<!-- CONTRIBUTING -->
## Results
### Symbol Accuracy
$\text{ASUR}$ (i.e., average symbols under repetitions) was adopted to compare symbol accuracy. 

$$ ASUR_k = \frac{1}{k} {\sum_{i=1}^{k}{C_i}} $$

where $C_i$ means the correctly recognized symbols in the $i$-th
repetition (there are 15 repetitions in total). $\text{ASUR}_{k}$ stands for the average correctly recognized
symbols per repetition when we take $k$ repetitions into account.
For more information about the $\text{ASUR}$, please refer to [ST-CapsNet: Linking Spatial and Temporal Attention with Capsule Network for P300 Detection Improvement](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10018278).
<div class="center">
<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">ASUR of II-A</th>
    <th colspan="3">ASUR of II-B</th>
    <th colspan="3">ASUR of IIb</th>
  </tr>
  <tr>
    <th>5</th>
    <th>10</th>
    <th>15</th>
    <th>5</th>
    <th>10</th>
    <th>15</th>
    <th>5</th>
    <th>10</th>
    <th>15</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ESVMs</td>
    <td>47.8</td>
    <td>64.6</td>
    <td>75.0</td>
    <td>65.0</td>
    <td>77.4</td>
    <td>83.7</td>
    <td>28.6</td>
    <td>30.2</td>
    <td>30.2</td>
  </tr>
  <tr>
    <td>CNN-1</td>
    <td>41.8</td>
    <td>60.0</td>
    <td>70.8</td>
    <td>58.6</td>
    <td>72.8</td>
    <td>78.9</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>MCNN-1</td>
    <td>42.8</td>
    <td>59.5</td>
    <td>70.5</td>
    <td>59.4</td>
    <td>73.7</td>
    <td>80.7</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>OLCNN</td>
    <td>50.8</td>
    <th>68.1</th>
    <th>77.3</th>
    <td>68.6</td>
    <td>80.2</td>
    <td>85.9</td>
    <td>29.0</td>
    <td>30.0</td>
    <td>30.3</td>
  </tr>
  <tr>
    <td>BN3</td>
    <th>51.8</th>
    <td>66.2</td>
    <td>75.4</td>
    <td>65.0</td>
    <td>77.1</td>
    <td>82.9</td>
    <td>26.2</td>
    <td>28.3</td>
    <td>29.2</td>
  </tr>
  <tr>
    <td>ERP-CapsNet</td>
    <td>45.8</td>
    <td>63.2</td>
    <td>73.7</td>
    <td>65.0</td>
    <td>77.1</td>
    <td>82.9</td>
    <td>28.4</td>
    <td>29.7</td>
    <td>30.1</td>
  </tr>
  <tr>
    <td>MsCNN-TL-ESVM</td>
    <td>43.6</td>
    <td>60.9</td>
    <td>71.3</td>
    <td>63.8</td>
    <td>77.6</td>
    <td>84.0</td>
    <td>29.4</td>
    <td>30.2</td>
    <td>30.4</td>
  </tr>
  <tr>
    <td>ST-CapsNet</td>
    <td>45.2</td>
    <td>64.4</td>
    <td>74.0</td>
    <td>66.2</td>
    <td>78.5</td>
    <td>84.3</td>
    <th>29.6</th>
    <th>30.3</th>
    <th>30.5</th>
  </tr>
  <tr>
    <td>EEGNet*</td>
    <td>45.6</td>
    <td>63.1</td>
    <td>73.3</td>
    <td>63.8</td>
    <td>78.2</td>
    <td>84.4</td>
    <td>28.6</td>
    <td>29.8</td>
    <td>30.2</td>
  </tr>
  <tr>
    <td>Improved EEGNet*</td>
    <td>48.2</td>
    <td>64.8</td>
    <td>75.4</td>
    <th>70.6</th>
    <td>81.6</td>
    <td>86.3</td>
    <td>28.2</td>
    <td>29.6</td>
    <td>30.1</td>
  </tr>
  <tr>
    <td>DeepConvNet*</td>
    <td>45.2</td>
    <td>63.6</td>
    <td>73.3</td>
    <td>70.4</td>
    <th>82.9</th>
    <th>88.1</th>
    <td>29.2</td>
    <td>30.1</td>
    <td>30.4</td>
  </tr>
</tbody>
</table>
</div>


Note

In the original papers (EEGNet, Improved EEGNet and DeepConvNet), the authors did not provide the results of P300 detection. I implemented them and ran those models to get the P300 detection results in BCI competitions II and III. The results ([in folder](result/without_fixed_seed)) of EEGNet, Improved EEGNet and DeepConvNet presented in this table were run without fixed seeds. The number of xdawn spatial filters of Improved EEGNet in this table was chosen to be 8.




### Effect of xdawn number on ASUR
Using [Improved EEGNet](https://journals.sagepub.com/doi/full/10.26599/BSA.2022.9050007), i.e. xdawn with EEGNet. Note, the results of the figure below were obtained without fixed seed.
![Xdawn + EEGNet](result/without_fixed_seed/Improved%20EEGNet/cat.png)



## Further reading
The softmax with temperature $t$ was used to get the probability of the model's output logits in this project. The larger $t$ is, the lower the confidence level of the model and the smoother the output.
$$p_{i} = \frac{exp(z_{i}/t)}{\sum_{j}{exp(z_{j}/t)}}$$
To show the effect of temperature to the symbol accuarcy, the Improved EEGNet(with 8 xdawn spatial filters) was run with fixed seed 0 on dataset II-A, II-B and IIb and the results were placed in [folder](/result/with_fixed_seed_0/).

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


## Citation
DeepConvNet and EEGNet are good enough
```sh
@article{wang2023review,
  title={A review of deep learning methods for cross-subject rapid serial visual presentation detection in World Robot Contest 2022},
  author={Wang, Zehui and Zhang, Hongfei and Ji, Zhouyu and Yang, Yuliang and Wang, Hongtao},
  journal={Brain},
  volume={9},
  number={2},
  pages={78--94},
  year={2023}
}
```

Improved EEGNet model:
```sh
@article{zhang2022improved,
  title={An improved EEGNet for single-trial EEG classification in rapid serial visual presentation task},
  author={Zhang, Hongfei and Wang, Zehui and Yu, Yinhu and Yin, Haojun and Chen, Chuangquan and Wang, Hongtao},
  journal={Brain Science Advances},
  volume={8},
  number={2},
  pages={111--126},
  year={2022},
  publisher={SAGE Publications Sage UK: London, England}
}
```
New comparison metric: $\text{ASUR}$
```sh
@ARTICLE{10018278,
  author={Wang, Zehui and Chen, Chuangquan and Li, Junhua and Wan, Feng and Sun, Yu and Wang, Hongtao},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
  title={ST-CapsNet: Linking Spatial and Temporal Attention with Capsule Network for P300 Detection Improvement}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TNSRE.2023.3237319}}
```

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III.svg?style=for-the-badge
[contributors-url]: https://github.com/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III.svg?style=for-the-badge
[forks-url]: https://github.com/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III/network/members
[stars-shield]: https://img.shields.io/github/stars/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III.svg?style=for-the-badge
[stars-url]: https://github.com/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III/stargazers
[issues-shield]: https://img.shields.io/github/issues/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III.svg?style=for-the-badge
[issues-url]: https://github.com/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III/issues
[license-shield]: https://img.shields.io/github/license/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III.svg?style=for-the-badge
[license-url]: https://github.com/wzhcoder/SOTA-of-P300-Detection-in-BCI-Competitions-II-III/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
