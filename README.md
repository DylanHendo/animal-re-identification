# QUT Final Year Thesis Project

## Background
As part of my engineering degree at the Queensland University of Technology, I was required to complete a final year thesis project, on an offered topic of my choosing, working with a supervisor at QUT. The topic I chose was ***Animal Re-Identification for Wildlife Monitoring***, under the supervision of Dr. Simon Denman.

## Data
Three seperate datasets were used for this project, to measure the performace of the seperate networks.
These 3 datasets were the following:
- [Amur Tiger Dataset](https://cvwc2019.github.io/challenge.html)
- [Chimpanzee Face Dataset](https://github.com/cvjena/chimpanzee_faces)
- [Humpback Whale Tail Dataset](https://www.kaggle.com/c/humpback-whale-identification/data)

## Instructions
The data can be downloaded from the 3 seperate links provided above. The images should then be converted to nump arrays, such that they can be fed trough the network. These arrays need to be converted to coloured images (3-channels).

My method was to save the numerical data as .npy files, such that they could be reloaded the same. If you wish to do it a different way, disregard the code for loading the images. All that is necessary is that you create an 'X' array and 'y' array, that can be split into testing, training and validation, at a split of 60/20/20.