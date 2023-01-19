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
    <img src="images/logo.png" alt="Logo" width="80" height="80">
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
This project aims to provide a simple general framework to help beginners who are struggling to achieve the state of the art of p300 detection in [BCI competitions II and III](https://www.bbci.de/competition/).


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.


### Prepare Datasets
1. Download [Data set IIb: ‹P300 speller paradigm›](https://www.bbci.de/competition/ii/) and [Data set II: ‹P300 speller paradigm›](https://www.bbci.de/competition/iii/). Dataset IIb only contains one subject, while dataset II has two subjects (A and B).
2. Download the true label of tests sets of [dataset IIa](https://www.bbci.de/competition/ii/results/labels_data_set_iib.txt) and the true label of tests sets of dataset II ( [A](https://www.bbci.de/competition/iii/results/albany/true_labels_a.txt) and [B](https://www.bbci.de/competition/iii/results/albany/true_labels_b.txt)). I have downloaded those labels of test sets and put them in the [folder](dataset_labels).
3. put those true labels into the root directory of the downloaded dataset, respectively.

### Requriments
- pytorch
- mne
- sklearn
- visdom


<!-- USAGE EXAMPLES -->
### Usage
1. Change src_path and root in main.py
2. run visdom in the console.
```sh
   visdom
```
3. Run main.py in the console.
```sh
   python main.py
```









<!-- CONTRIBUTING -->
## Results
![Xdawn + EEGNet](images/orignial/cat.png)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


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
