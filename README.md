# DLGI
Two simple network (CNN and DNN) are provided to increase the quality of image reconstruction for computational ghost imaging.

# Requirements: 

> Anaconda 3; Python 3.8.13; Pytorch 1.7.0.

# How to use:

Step 1: create a virtual environment
```
conda create -n DLGI
conda activate DLGI
```

Step 2: download the required packages
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch (cuda version should be chosen according to your GPU.)
conda install jupyter
pip install matplotlib
pip install scikit-image
pip install opencv-python
```

Step 3: download and extract the ZIP file.

Step 4: Run train.ipynb in DNN to train a deep neural network and test.ipynb in DNN to test the trained network.

Step 5: Run train.ipynb in CNN to train a convolutional neural network and test.ipynb in CNN to test the trained network.
