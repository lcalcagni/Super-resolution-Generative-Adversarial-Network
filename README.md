# Super-Resolution Generative Adversarial Network (SRGAN) with Keras

This is a Keras & TensorFlow implementation of the SRGAN proposed in the scientific article[ \[1\]](https://arxiv.org/pdf/1609.04802.pdf).


## Installation

The requirements are:

 Python 3.6\
 requirements.txt

1) First, download the repository and install the requirements:
```
git clone https://github.com/lcalcagni/Super-resolution-Generative-Adversarial-Network
cd Super-resolution-Generative-Adversarial-Network/
pip install -r requirements.txt
```


## Get Dataset

2) Download the images for training and save them into input/data_train and some of them into input/data_test for testing.


## Train & Predict

3) To train, configure **main.py** with the desired number of **epochs**, **batch_size** and set **mode = 'train'**. Then, execute:
```
python main.py
```

4) To predict, configure **main.py** with the desired number of **samples** and set **mode = 'predict'**. Then, execute:
```
python main.py
```

---  

### References
1. C. Ledig, L. Theis, F. Huszár, J. A. Caballero, A. Aitken, A. Tejani, J. Totz, Z. Wang, and W. Shi. -*Photo-realistic single image super-resolution using a generative adversarial network*. **2017** IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 105–114, 2016.

3.  I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. *Generative adversarial nets*. In Advances in Neural Information Processing Systems (NIPS), pages 2672–2680, **2014**.
