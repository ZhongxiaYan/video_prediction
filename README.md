# Hierarchical Video Prediction

Zhongxia Yan, Jeffrey Zhang

We implemented a generative model to predict future frames of a human action video hierarchically. This repo contains all of the infrastructure and logic for our project. Our models extends off the model described in [Learning to Generate Long-term Future via Hierarchical Prediction](https://sites.google.com/a/umich.edu/rubenevillegas/hierch_vid). We independently explored many ways to improve the model and document the explorations here.

## Summary
Video prediction involves predicting T future frames of a video given k original frames of the video. Our hierarchical approach predicts **human action** videos by taking the pose (location of human joints) in the k original frames (we can estimate the pose using methods such as the [Hourglass Network](https://arxiv.org/abs/1603.06937)), predicting the pose in the next T frames, then predicting T frames that correspond to the T poses and also match the k original frames.

![alt text](../resources/model_overview.png)

## Dataset
We train and evaluate our models on the [Penn Action Dataset](http://dreamdragon.github.io/PennAction/). Below are examples of an image and a pose.

![alt text](../resources/penn_action_image_ex.png)
![alt text](../resources/penn_action_heatmap_ex.png)

## Method
### LSTM

![alt text](../resources/lstm.png)

There are 13 (x, y) coordinates per pose. For the first k frames, we feed the 13 (x, y) coordinate values of the joints into the LSTM and iteratively generate the hidden states. For the next T frames, we feed in **0** and record the hidden state outputs from t = k + 1 to t = k + T. We use two fully connected layers on top of each hidden state vector to predict 13 (x, y) coordinates for the pose.

![alt text](../resources/lstm_equations.png)

### Analogy Network

![alt text](../resources/analogy.png)
Our analogy network is a generative model that takes in the video frame at time t1, pose at time t1, and the pose at time t2 to predict the video frame at time t2. In our model, f_img and f_pose are both encoders implemented as the first c convolution layers of VGG (c is a parameter that we explore), and f_dec is the decoder implemented with deconvolutions mirroring the VGG convolutions of the encoders. Here our pose are gaussian heatmaps with the joint (x, y) coordinates as centers (g represents the gaussian operator).

![alt text](../resources/analogy_generator_equation.png)
![alt text](../resources/analogy_network_generator.png)

Our loss function has three parts.

![alt text](../resources/analogy_losses.png)

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

