#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP A4 2023
parser_model.py: Feed-Forward Neural Network for Dependency Parsing
Authors: Sahil Chopra, Haoshen Hong, Nathan Schneider, Lucia Donatelli
"""
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParserModel(nn.Module):
    """ Feedforward neural network with an embedding layer and two hidden layers.
    The ParserModel will predict which transition should be applied to a
    given partial parse configuration.

    PyTorch Notes:
        - Note that "ParserModel" is a subclass of the "nn.Module" class. In PyTorch all neural networks
            are a subclass of this "nn.Module".
        - The "__init__" method is where you define all the layers and parameters
            (embedding layers, linear layers, dropout layers, etc.).
        - "__init__" gets automatically called when you create a new instance of your class, e.g.
            when you write "m = ParserModel()".
        - Other methods of ParserModel can access variables that have "self." prefix. Thus,
            you should add the "self." prefix layers, values, etc. that you want to utilize
            in other ParserModel methods.
        - For further documentation on "nn.Module" please see https://pytorch.org/docs/stable/nn.html.
    """
    def __init__(self, embeddings, n_features=36,
        hidden_size=200, n_classes=3, dropout_prob=0.5):
        """ Initialize the parser model.

        @param embeddings (ndarray): word embeddings (num_words, embedding_size)
        @param n_features (int): number of input features
        @param hidden_size (int): number of hidden units
        @param n_classes (int): number of output classes
        @param dropout_prob (float): dropout probability
        """
        super(ParserModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.embeddings = nn.Parameter(torch.tensor(embeddings))

        ### CODING ASSIGNMENT 1
        ### YOUR CODE HERE (~9-10 Lines)
        ### TODO:
        ###     1) Declare `self.embed_to_hidden_weight` and `self.embed_to_hidden_bias` as `nn.Parameter`.
        ###        Initialize weight with the `nn.init.xavier_uniform_` function and bias with `nn.init.uniform_`
        ###        with default parameters.
        ###     2) Construct `self.dropout` layer.
        ###     3) Declare `self.hidden_to_logits_weight` and `self.hidden_to_logits_bias` as `nn.Parameter`.
        ###        Initialize weight with the `nn.init.xavier_uniform_` function and bias with `nn.init.uniform_`
        ###        with default parameters.
        ###
        ### Note: Trainable variables are declared as `nn.Parameter` which is a commonly used API
        ###       to include a tensor into a computational graph to support updating w.r.t its gradient.
        ###       Here, we use Xavier Uniform Initialization for our Weight initialization.
        ###       It has been shown empirically, that this provides better initial weights
        ###       for training networks than random uniform initialization.
        ###       For more details checkout this great blogpost:
        ###             http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        ###
        ### Please see the following docs for support:
        ###     nn.Parameter: https://pytorch.org/docs/stable/nn.html#parameters
        ###     Initialization: https://pytorch.org/docs/stable/nn.init.html
        ###     Dropout: https://pytorch.org/docs/stable/nn.html#dropout-layers
        ### 
        ### See the PDF for hints.

        # Declare `self.embed_to_hidden_weight` and `self.embed_to_hidden_bias` as `nn.Parameter`
        # Initialize weights & bias parameters with `nn.init.xavier_uniform_` and bias with `nn.init.uniform_`
        # Weight & bias parameters connect embedding layer to hidden layer 
        self.embed_to_hidden_weight = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(hidden_size, n_features * self.embed_size)))#tensor initialized with values from Xavier uniform distribution
        self.embed_to_hidden_bias = nn.Parameter(torch.nn.init.uniform_(torch.empty(hidden_size))) #tensor initialized with values drawn from a uniform distribution

        # Construct `self.dropout` layer
        # Dropout layer randomly drops units with a probability of dropout_prob during training to prevent overfitting
        self.dropout = nn.Dropout(dropout_prob)

        # Declare `self.hidden_to_logits_weight` and `self.hidden_to_logits_bias` as `nn.Parameter`
        # Initialize weight with `nn.init.xavier_uniform_` and bias with `nn.init.uniform_`
        self.hidden_to_logits_weight = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(hidden_size, n_classes)))
        self.hidden_to_logits_bias = nn.Parameter(torch.nn.init.uniform_(torch.empty(n_classes)))

        ### END YOUR CODE

    def embedding_lookup(self, w):
        """ Utilize `w` to select embeddings from embedding matrix `self.embeddings`
            @param w (Tensor): input tensor of word indices (batch_size, n_features)

            @return x (Tensor): tensor of embeddings for words represented in w
                                (batch_size, n_features * embed_size)
        """
        ### CODING ASSIGNMENT 2
        ### YOUR CODE HERE (~1-4 Lines)
        ### TODO:
        ###     1) For each index `i` in `w`, select `i`th vector from self.embeddings
        ###     2) Reshape the tensor using `view` function if necessary
        ###
        ### Note: All embedding vectors are stacked and stored as a matrix. The model receives
        ###       a list of indices representing a sequence of words, then it calls this lookup
        ###       function to map indices to sequence of embeddings.
        ###
        ###       This problem aims to test your understanding of embedding lookup,
        ###       so DO NOT use any high level API like nn.Embedding
        ###       (we are asking you to implement that!). Pay attention to tensor shapes
        ###       and reshape if necessary. Make sure you know each tensor's shape before you run the code!
        ###
        ### Pytorch has some useful APIs for you, and you can use either one
        ### in this problem (except nn.Embedding). These docs might be helpful:
        ###     Index select: https://pytorch.org/docs/stable/torch.html#torch.index_select
        ###     Gather: https://pytorch.org/docs/stable/torch.html#torch.gather
        ###     View: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Flatten: https://pytorch.org/docs/stable/generated/torch.flatten.html

        # Get the word embeddings for the given word indices
        x = self.embeddings[w] #w is an input tensor representing word indices with shape (batch_size, n_features)

        if x.dim() == 2:#check if embeddings are not flattened yet 
           x = x.view(x.size(0), -1)#if x has a dimension of 2, it means that n_features is greater than 1, and the embeddings need to be reshaped to have a shape of (batch_size, n_features * embed_size)


        # Flatten the input tensor w
        #w_flat = w.view(w.size(0), -1)  # shape: (batch_size, n_features * embed_size)

        # Get the word embeddings for the given word indices
        #x = torch.matmul(w_flat, self.embed_to_hidden_weight.t()) + self.embed_to_hidden_bias


         # shape: (batch_size, hidden_size)

        ### END YOUR CODE
        return x


    def forward(self, w):
        """ Run the model forward.

            Note that we will not apply the softmax function here because it is included in the loss function nn.CrossEntropyLoss

            PyTorch Notes:
                - Every nn.Module object (PyTorch model) has a `forward` function.
                - When you apply your nn.Module to an input tensor `w` this function is applied to the tensor.
                    For example, if you created an instance of your ParserModel and applied it to some `w` as follows,
                    the `forward` function would called on `w` and the result would be stored in the `output` variable:
                        model = ParserModel()
                        output = model(w) # this calls the forward function
                - For more details checkout: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.forward

        @param w (Tensor): input tensor of tokens (batch_size, n_features)

        @return logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 without applying softmax (batch_size, n_classes)
        """
        ### CODING ASSIGNMENT 3
        ### YOUR CODE HERE (~3-5 lines)
        ### TODO:
        ###     Complete the forward computation as described in write-up. In addition, include a dropout layer
        ###     as decleared in `__init__` after ReLU function.
        ###
        ### Note: We do not apply the softmax to the logits here, because
        ### the loss function (torch.nn.CrossEntropyLoss) applies it more efficiently.
        ###
        ### Please see the following docs for support:
        ###     Matrix product: https://pytorch.org/docs/stable/torch.html#torch.matmul
        ###     ReLU: https://pytorch.org/docs/stable/nn.html?highlight=relu#torch.nn.functional.relu


        # Embedding lookup
        x = self.embedding_lookup(w)#retrieve embeddings for lookup tokens (batch_size, n_features * embed_size)

        # Reshape x to match the dimensions of embed_to_hidden_weight
        x = x.view(x.size(0), self.n_features * self.embed_size)

        # Linear transformation 1
        hidden = torch.matmul(x, self.embed_to_hidden_weight.t()) + self.embed_to_hidden_bias#linear transformation between x and self.embed_to_hidden_weight followed by bias
        # Resulting tensor hidden has a shape of (batch_size, hidden_size)

        # ReLU activation
        hidden = F.relu(hidden)#introduce nonlinearity to model, tensor shape is still (batch_size, hidden_size)

        # Dropout layer
        hidden = self.dropout(hidden)#applied to hidden layer to prevent overfitting 

        # Linear transformation 2
        # Apply matrix multiplication between the hidden layer activations and the weight matrix, followed by an addition of the bias vector
        # Resulting tensor logits represents the predictions of the model without applying the softmax function
        # It has a shape of (batch_size, n_classes), where n_classes is the number of output classes.
        #logits = torch.matmul(hidden, self.hidden_to_logits_weight) + self.hidden_to_logits_bias
        logits = torch.matmul(hidden, self.hidden_to_logits_weight) + self.hidden_to_logits_bias

         #logits used for loss function

        ### END YOUR CODE
        return logits

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simple sanity check for parser_model.py')
    parser.add_argument('-e', '--embedding', action='store_true', help='sanity check for embeding_lookup function')
    parser.add_argument('-f', '--forward', action='store_true', help='sanity check for forward function')
    args = parser.parse_args()

    embeddings = np.zeros((100, 30), dtype=np.float32)
    model = ParserModel(embeddings)

    def check_embedding():
        inds = torch.randint(0, 100, (4, 36), dtype=torch.long)
        selected = model.embedding_lookup(inds)
        assert np.all(selected.data.numpy() == 0), "The result of embedding lookup: " \
                                      + repr(selected) + " contains non-zero elements."

    def check_forward():
        inputs =torch.randint(0, 100, (4, 36), dtype=torch.long)
        out = model(inputs)
        expected_out_shape = (4, 3)
        assert out.shape == expected_out_shape, "The result shape of forward is: " + repr(out.shape) + \
                                                " which doesn't match expected " + repr(expected_out_shape)

    if args.embedding:
        check_embedding()
        print("Embedding_lookup sanity check passes!")

    if args.forward:
        check_forward()
        print("Forward sanity check passes!")