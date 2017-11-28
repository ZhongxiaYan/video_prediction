# Hierarchical Video Prediction

Code reimplmenetation of Learning to Generate Long-term Future via Hierarchical Prediction! This github repo contains working code for Villegas et al's paper on Learning to Generate Long-term Future via Hierarchical Prediction. This README displays results produced from our models, our own experimentation with the method, and how to run the code.

## Method

Insert diagrams and images from our write up.
Include hyper parameters we used.
Original details can be found at https://arxiv.org/abs/1704.05831.

## Results
Results for each action on Penn Action Dataset 

|   Baseball Pitch   |  Baseball Swing  | Bench Press | Bowl |
| ------------- |:-------------:|:-----:|:-----:|
| ![alt text][pitch]  | ![alt text_2][swing] | ![alt text_3][bench]| ![alt text_4][bowl] |

|   Clean and Jerk  |  Golf Swing  | Jump Rope | Jumping Jacks |
| ------------- |:-------------:|:-----:|:-----:|
| ![alt text][pitch]  | ![alt text_2][pitch] | ![alt text_3][pitch]| ![alt text_4][pitch] |

|   Baseball Pitch   |  Baseball Swing  | Bench Press | Bowl |
| ------------- |:-------------:|:-----:|:-----:|
| ![alt text][pitch]  | ![alt text_2][pitch] | ![alt text_3][pitch]| ![alt text_4][pitch] |


[pitch]: https://github.com/ZhongxiaYan/video_prediction/blob/master/src/0050.gif 
[swing]: https://github.com/ZhongxiaYan/video_prediction/blob/master/src/0280.gif 
[bench]: https://github.com/ZhongxiaYan/video_prediction/blob/master/src/0441.gif 
[bowl]: https://github.com/ZhongxiaYan/video_prediction/blob/master/src/0544.gif 

## Experimentation
### Truncated VGG convolution layers

### Full Image vs Crop

### Conv5 vs Conv5 + Residual 

### Checkboarding and Deconvolution Artifacts

## How to Run

---
## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/ZhongxiaYan/video_prediction/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ZhongxiaYan/video_prediction/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.


