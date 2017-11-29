# Hierarchical Video Prediction

Zhongxia Yan, Jeffrey Zhang

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
| ![alt text][clean]  | ![alt text_2][golf] | ![alt text_3][jump]| ![alt text_4][jacks] |

|   Pullup  |  Pushup  | Situp | Squat |
| ------------- |:-------------:|:-----:|:-----:|
| ![alt text][pullup]  | ![alt text_2][pushup] | ![alt text_3][situp]| ![alt text_4][squat] |


[pitch]: https://github.com/ZhongxiaYan/video_prediction/blob/master/gifs/pitch.gif 
[swing]: https://github.com/ZhongxiaYan/video_prediction/blob/master/gifs/swing.gif 
[bench]: https://github.com/ZhongxiaYan/video_prediction/blob/master/gifs/bench.gif 
[bowl]: https://github.com/ZhongxiaYan/video_prediction/blob/master/gifs/bowl.gif 
[clean]: https://github.com/ZhongxiaYan/video_prediction/blob/master/gifs/clean.gif 
[golf]: https://github.com/ZhongxiaYan/video_prediction/blob/master/gifs/golf.gif 
[jump]: https://github.com/ZhongxiaYan/video_prediction/blob/master/gifs/jump.gif 
[jacks]: https://github.com/ZhongxiaYan/video_prediction/blob/master/gifs/jacks.gif 
[pullup]: https://github.com/ZhongxiaYan/video_prediction/blob/master/gifs/pullup.gif 
[pushup]: https://github.com/ZhongxiaYan/video_prediction/blob/master/gifs/pushup.gif 
[situp]: https://github.com/ZhongxiaYan/video_prediction/blob/master/gifs/situp.gif 
[squat]: https://github.com/ZhongxiaYan/video_prediction/blob/master/gifs/squat.gif 

## Experimentation
### Truncated VGG convolution layers

### Full Image vs Crop

### Conv5 vs Conv5 + Residual 

### Checkboarding and Deconvolution Artifacts

## How to Run
```markdown
python run2.py --model villegas_combined --config rgb_L_13_cropped_consistent 
```

