# Video_NN
This page is my attempt to study image processing using neural networks, namely, to learn how to predict the average frame in a video by two extremes.
Below is a description of the README: <br />
[Structure of repository](#Structure-of-repository) <br />
[Problem with MSE-loss and solution](#Problem-with-MSE-loss-and-solution) -  an interesting and frequently encountered problem for me is the blurring of the generated image and the solution to this issue. <br />
[Training](#Training) - how I trained the neural network. <br />
[Сonclusion and future plans](#Сonclusion-and-future-plans) - things that I would like to explore further and improvements that I would like to add. 
# Structure of repository:
1. [VNN.py](https://github.com/berkutivan/Video_NN/blob/main/VNN.py)- all functions of Unet
2. [main.py](https://github.com/berkutivan/Video_NN/blob/main/main.py) - code to generate the better quality video
3. [video_functions.py](https://github.com/berkutivan/Video_NN/blob/main/video_functions.py) - auxiliary code for video processing
4. [training_file.ipynb](https://github.com/berkutivan/Video_NN/blob/main/training_file.ipynb) - notebook with neural network training
# U-net scheme:
U-net is a convolutional neural network that was invented for image segmentation. This is how her scheme looks like:
 <br />
![U-net scheme](https://aswinvisva.me/images/unet.png)
 <br />
[Where can I read more about U-net](https://www.geeksforgeeks.org/u-net-architecture-explained/). This architecture is good at detecting images and is often used in tasks where you need to compare incoming images or identify some patterns: be it light highlights or folds on clothes. 
I chose U-net because I assumed that the task of prolongate the video involves generating difficult objects such as ripples of water, sparks, falling snow or rain, which in turn is a pattern for similar structures.  <br /> 
The problem with U-net in this task is that this model has a lot of weights, which complicates the generation of frames in real time.
My model includes ~ 3 million weights, unfortunately, I could not test it on a good graphics card, but the speed of generating RGB images with dimensions of 448X224 pixels on the built-in AMD RADEON TM graphics card was about 5 seconds.
# Problem with MSE-loss and solution: 
First, I trained the model using MSE loss, which led (after 30 epochs) to the following result:
![Generate img with MSE loss](https://github.com/berkutivan/Video_NN/blob/main/before_result.jpg)
Explanation: A time series of 3 images was extracted from the dataset. The first and second picture were fed to the input of the neural network, and the resulting one (marked NN) was compared with the second one. <br />
The problem is that the generated and original images are compared pixel by pixel, which is inconvenient in this task. 
A small example: we have to generate an image of the chessboard. Even if we get a realistic image of the chessboard, but shifted by one square relative to the real one, the MSE loss will be maximum. As a result, after several epochs, our neural network will produce just a gray picture. <br />
The solution is this: compare the images not pixel-by-pixel, but as if a person compared them figuratively. For me, fortunately or unfortunately, this problem has already been solved.
Images are collapsed using a model that has already been pre-trained and is able to recognize images. After that, MSE-loss is applied to the convolutions.
I used VGG16 as such a model.  I got the following results:
![GEnerate img with new loss](https://github.com/berkutivan/Video_NN/blob/main/after_result.jpg)

# Training
# Сonclusion and future plans
Thus, we have disassembled the U-net device, methods of dealing with blurred images and technical aspects of my research.
In conclusion, I want to say that this topic is more a field for research, for example, you can pre-evaluate whether the generated frame will turn out to be of high quality and based on this decide whether it is worth spending time creating it. For example, in a video there is a transition between scenes or a camera view. In this case, it doesn't make much sense to generate a frame in the middle. And this is just one of the many things that I would like to add in the future.<br />
small prolongate video:
![small prolongate video](https://github.com/berkutivan/Video_NN/blob/main/generate.gif)
