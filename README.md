# chromosome_segementation

### Introduction

The repo is aimed for segmentation of overlapping chromosomes, as described in the problem statement given on [AI ON website](http://ai-on.org/projects/visual-segmentation-of-chromosomal-preparations.html)

The repo uses [U-Net](https://arxiv.org/abs/1505.04597), state-of-the art segmentation net for segmenting overlapping chromosomes. The repo used Lasagne, a Theano based library for segmentation.

### Methodology

The data consists of 4 classes, where class 4 is the common region between 2 overlapping chromosomes. The classes 1 & 2 , are non-overlapping part of each of the chromosomes. Class 0 is the background

The performance of the net was observed using mean_dic_score. It was computed as 'dice_score = 2*I/(GT + PL)' where I is the sum of the number of pixels predicted correctly except background, GT is the number of pixels which belong to ground-truth except background and PL is the number of pixels in predicted image except background.

There were 2 methods of training attempted
- Treating all the classes independently (param combine_label = True in segmentation.py)
- Treating Class 1 & Class 2 as same i.e. Class 1 and Class 3 as Class 2 (param combine_label = False in segmentation.py)Assumption being the non-overlapping parts inherently aren't different in each chromosomes

The training log with combined_label looks like this
![image_train](/images/combined_label_train.png)


### Results

With combined labels, could reach a dice score as high as 0.97. Some the predictions look like this
![predict_combined](/images/Vis_combined_label_2.png)
![predict_combined](/images/Vis_combined_label_1.png)
