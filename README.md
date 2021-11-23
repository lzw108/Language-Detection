Language Detection

# Language Detection

This is a project on language detection.

 
## Model
The model is based on the TextCNN [<sup>[1]</sup>](#refer-anchor-1)[<sup>[2]</sup>](#refer-anchor-2).
<div align="center">
<img src="https://i.loli.net/2021/11/22/JhUq6nzjiFXItos.png" height="800" width="650" >
 </div>
 
## Requirements
- python 3.7
- Pytorch 1.5
- CUDA (Recommended version >=10.0)
## Getting Started
## Data
We need download [Tatoeba dataset](https://tatoeba.org/eng/downloads) as our train data, which
 includes 403 kinds of language. You can download in data directory:
```python
wget http://downloads.tatoeba.org/exports/sentences.tar.bz2
bunzip2 sentences.tar.bz2
tar xvf sentences.tar
```
# Train
First, we should process data:
```python
python main.py --data_process 
```
It will first call data_process.py to split data to train_process.csv and test_process.csv.
Before training, you can adjust the parameters in args.py. Then you can train the model:
```python
python main.py --train 
```
You will get the model in the model Directory and a vocab in the data directory,
## Test
You can use test_process.csv to test the ability of the model you have trained:
```python
python main.py --test
```
The model I trained has achieved 95.38% accuracy in the test set.
## Test any single sentence
```python
python main.py --test_single
```
Then you can follow the prompts to enter a single sentence and enjoy it.

## Improvement
At present, it only randomly initializes the word vector. In the future, you can try to load the word vector trained by FastText [<sup>[3]</sup>](#refer-anchor-3)[<sup>[4]</sup>](#refer-anchor-4)or rewrite FastText.

- [1] [Chen Y. Convolutional neural network for sentence classification.University of Waterloo, 2015.](https://arxiv.org/pdf/1408.5882.pdf)
- [2] [Zhang Y, Wallace B. A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification. arXiv preprint arXiv:1510.03820, 2015.](https://arxiv.org/pdf/1510.03820.pdf)
- [2] [Joulin A, Grave E, Bojanowski P, et al. Bag of tricks for efficient text classification[J]. arXiv preprint arXiv:1607.01759, 2016.](https://arxiv.org/abs/1607.01759)
- [3] [Joulin A, Grave E, Bojanowski P, et al. Bag of tricks for efficient text classification. arXiv preprint arXiv:1607.01759, 2016.](https://arxiv.org/abs/1612.03651)
