# A python implementation of GloVe

In this project, I will try to build a toy model in Python according to the original GloVe paper. The main purpose is to help understand throughly the mechanism of the algorithme and implement the model in a smaller scale dataset to analyse its performance.
 
The original paper: https://nlp.stanford.edu/pubs/glove.pdf

## 1. What’s word embedding:

Word embedding is a technique which allows us to transforme words into vectors so that the data could be fed to ML algorithms. The resulting vectors could no only help computer to compute the similitude of a word to another, but also contain useful semantic and synthetic information about the words. The most common word embedding techniques are Word2Vec and GloVe. Word2Vec defining a fake task: to predict the surrounding words given a target word or to predict the surrounding words using the target word. The weight of the word in the hidden layer after training is the word vector. However, this method concentrates on the local relationship of words and dose not make use of the global statistic of the words. 

## 2. What’s GloVe:

> *You shall know a word by the company it keeps. - John Rupert Firth, Linguist*

The basic information all unsupervised algorithms use is the statistic of word occurrence, different models use it in different ways. The main difference of GloVe compared to Word2Vec is the use of co-occurence matrix. It allows the global corpus statistics to be captured by the model.


2. 1 **Co-occurence matrix**

The matrix of word-word co-occurrence tabulates the number of times word A occurs in the context of word B.

In the original paper, a decreasing weighting function is used so that word pairs that are d words apart contribute 1/d to the total count. The window size and the decision of whether distinguishing left context from right context should also be made before constructing the co-occurrence matrix. The context window size used in the paper is 10.

(Example) Consider the sentence: “Your model is only as good as your data”. The co-occurence matrix with context window 2 without distinction of left and right context is:



2. 2 **Notations**

Let the corpus be denoted by $C$, and its vocabulary size (the number of distinct words) be denoted by $V$.

Let the matrix of word-word co-occurence counts be denoted by $X$, with $X_{ij}$ = the number of times word $j$ occurs in the context of word $i$.

Each word $i$ of the vocabulary is associated with two weight vectors $w_i, \hat{w}_i$ of dimension D, and two biais $b_i, \hat{b}_i$ of length 1. 

The final embedding of the word $i$ is the sum of $w_i$ and $\hat{w}_i$. 



2. 3 **Loss function**

Given a co-occurence matrix, we could compute the probability the word *j* appears in the context of the word *i*:

$$P_{ij}=P(j|i)=X_{ij}/X_i$$

To extract meaning from the co-occurence matrix, the authors use the ratio of probabilities instead of the raw probabilities.

Here is the classic example explaining the idea. Given two main words $i=ice$ and $j=steam$ and a third one $k$, we obtained the following statistics from a co-occurence matrix:

<p align="center">
<img width="400" alt="Screenshot 2023-01-14 at 17 03 48" src="https://user-images.githubusercontent.com/107317997/212481630-fbc83b4b-9cc1-4174-b014-9bb5a65cee78.png">
</p>

Word closer to $i=ice$ has a ratio much greater than 1 and word closer to $j=steam$ has a ratio much lower than 1. Words that are related or unrelated to both $i$ and $j$ should have a ratio around 1.

To apply this idea in the training of word vectors, we suppose that a function that allows us to compute the ratio exists given the word $i$, $j$ and $k$.

$$F(w_i,w_j,w_k)=\frac{P_{ik}}{P_{jk}}$$

By imposing conditions to the function and the word vectors, we could obtain a relationship between two word vectors (More details could be found in the original paper). Based on the relationship, the loss function is defined as :

$$J=\sum_{i,j=1}^{V}f(X_{ij})(w_i^T\hat{w_j}+b_i+\hat{b_j}-log(X_{ij}))^2$$


2. 4 **Weight function**

Weight function is introduced to avoid weighing all co-occurence equally. 

$$f(x)=1_{x<x_{max}}(x)(x/x_{max})^\alpha+1_{x>=x_{max}}(x)$$


In the paper, $x_{max}$ is fixed to 100 and alpha to 3/4.
The result is a function like this:

<p align="center">
<img width="297" alt="Screenshot 2023-01-14 at 17 13 40" src="https://user-images.githubusercontent.com/107317997/212482513-1f00de60-c1c3-4341-97ca-4927f973a6da.png">
</p>

2. 5 **Gradients**

The value of loss function is based on four variables: W1, W2, B1, B2. The gradient pf the loss function contains the derivatives of J based of each variable 
￼
$$\frac{\partial J}{\partial w_i}=\sum_{j=1}^{V}2f(X_{ij})(w_i^T\hat{w_j}+b_i+\hat{b_j}-log(X_{ij}))\hat{w_j}$$

$$\frac{\partial J}{\partial b_i}=\sum_{j=1}^{V}2f(X_{ij})(w_i^T\hat{w_j}+b_i+\hat{b_j}-log(X_{ij}))$$
