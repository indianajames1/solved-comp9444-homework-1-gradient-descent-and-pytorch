Download Link: https://assignmentchef.com/product/solved-comp9444-homework-1-gradient-descent-and-pytorch
<br>
Part 1

For Part 1 of the assignment, you should work through the file part1.py and add functions where specified.

<h1>Part 2</h1>

For Part 2, you will develop a linear model to solve a binary classification task on two dimensional data. The file <a href="https://www.cse.unsw.edu.au/~cs9444/19T3/hw1/hw1/data/">data/binar</a><a href="https://www.cse.unsw.edu.au/~cs9444/19T3/hw1/hw1/data/">y</a><a href="https://www.cse.unsw.edu.au/~cs9444/19T3/hw1/hw1/data/">_classification_data.</a><a href="https://www.cse.unsw.edu.au/~cs9444/19T3/hw1/hw1/data/">p</a><a href="https://www.cse.unsw.edu.au/~cs9444/19T3/hw1/hw1/data/">kl</a> contains the data for this part. We have included the file used to generate the data as data_generator.py. You may examine this for your reference, or modify it if you wish to watch Gradient Decent take place on different data. Note that running this file will replace the pickle file with another stochastically generated dataset. This shouldn’t cause your solution to fail, but it will cause the final output image to appear different. It is good to check that your file works with the original pickle file provided.

The file part2.py is the one you need to modify. It contains a skeleton definition for the custom LinearModel class. You need to complete the appropriate functions in this class.

You may modify the plotting method during development (LinearModel.plot()) – it may help you to visualize additional information. Prior to submission, however, verify that the expected output is being produced with the original, unaltered, code.

When completed, a correct implementation should produce the following image, along with model accuracies at each training step printed to stdout:

Example output from a correctly implemented Part 2.

This shows the provided datapoints, along with the decision boundary produced by your model at each step during training (dotted green lines). You can see that the data is not linearly separable, however the optimal separating plane is still found. For this data and model, it is impossible to achieve 100% accuracy, and here only 92% or 94% is achieved (with one point lying very close to th boundary).

<h2>Task 1 – Activation Function</h2>

Implement a sigmoid activation function. It is good practice when developing with deep learning models to constrain your code as much as possible, as the majority of errors will be silent and it is very easy to introduce bugs. Passing incorrectly shaped tensors into a matrix multiplication, or example, will not appear as on error, but will instead broadcast. For this reason, you must ensure that the activation method raises a ValueError with an appropriate error message if a list, boolean, or numpy array is passed as input. Ensure that singular numpy types (such as numpy.float64) can be handled.

Weights and other variables should be implemented as numpy arrays, not lists. This is good practice in general when the size of a sequence is fixed.

<h2>Task 2 – Forward Pass</h2>

Implement the forward pass of the model following the structure specified. In other words, given an input, return the output of the model.

Task 3 – Loss

Implement the cross entropy loss function for the learning algorithm to minimize. See function docstring for more information.

Task 4 – Error

Implement an error function to return the difference between target and actual output

<h2>Task 5 – Backward Pass</h2>

Here you are required to implement gradient descent without using pytorch or autograd. Although this is difficult in general, we have tried to make it easier in this case by sticking to a single-layer network and making use of other simplifications (see function docstring for details).

<h1>Part 3</h1>

Here you will be implementing networks to recognize handwritten Hiragana symbols. The dataset to be used is Kuzushiji-MNIST or KMNIST for short. The paper describing the dataset is available <a href="https://arxiv.org/pdf/1812.01718.pdf">here</a>. It is worth reading, but in short: significant changes occurred to the language when Japan reformed their education system in 1868, and the majority of Japanese today cannot read texts published over 150 years ago. This paper presents a dataset of handwritten, labeled examples of this old-style script (Kuzushiji). Along with this dataset, however, they also provide a much simpler one, containing 10 Hiragana characters with 7000 samples per class. This is the dataset we will be using.

Text from 1772 (left) compared to 1900 showing the standardization of written Japanese.

A large amount of code has been provided for you. You should spend time understanding this code. A simple model has also been provided for your reference that should make the other tasks easier. It is a good idea to use the same structure provided in this model in the code you write. The model is a linear model very similar to what you implemented in Part 1, with all inputs mapped directly to 10 ReLU activated nodes. Note that it is not identical to the model in Part 1 – do not try to reverse engineer Part 1 from this model. Technically the activation function here is redundant – however we have included it as an example of how to make use of torch.nn.functional.

When run, part3.py will train three models (one provided, two you will implement), a Linear Network, Feed Forward network, and a Convolutional Network, for 10 epochs each. A full run of part3.py can take up to an hour – however during development it is a good idea to train for fewer epochs initially, until you observe roughly correct behaviour.

A correct run over all epochs should produce the following plot:

Output plot for Part 3. On this dataset, learning occurs very fast, with a large amount occurring in one epoch. The increasing capacity and corresponding performance of each network type is clearly visible.

<h1>Contraints</h1>

<ol>

 <li>Do not use nn.Sequential, instead use torch.nn.functional to setup your network. An example of a linear net is present.</li>

 <li>In this assignment, all code will run on a CPU, regardless of which version of pytorch is installed. You may set code to run on a GPU during development if you wish to speed up training (although this wont make a big difference for this assignment), but ensure you do not have .cuda() or .to() calls in the code you submit.</li>

 <li>Shuffling in the Dataloader has been set to off for testing purposes – in practice this would be set to True. Do not modify this.</li>

 <li>Do not modify the training and testing code (exception: you may wish to comment out the code displaying the sample images.</li>

</ol>

This code is marked with the comment # Can comment the below out during development ).

<ol start="5">

 <li>Do not change the names of files.</li>

 <li>Naming: Standard convention is to name fully connected layers fc1, fc2 etc, where the number indicates depth. Similarly for convolutional layers, conv1, conv2 should be used. Task 1 – View Batch</li>

</ol>

Whenever developing deep learning models, it is absolutely critical to begin with a complete understanding of the data you are using. For this reason, implement a function that returns an 8×8 tiling of a batch of 64 images produced by one of the dataloaders, and the corresponding labels in a numpy array. Once implemented correctly, you should see he image shown below when running part3.py.

First batch of images from KMNIST tiled in 8×8 grid, produced by a correct view_batch

You should also see the following printed to stdout:

[[8 7 0 1 4 2 4 8]

[1 1 5 1 0 5 7 6]  [1 7 9 5 7 3 7 5]  [6 6 2 7 6 0 9 6]  [1 5 9 5 8 0 0 8]

[8 6 7 7 7 8 1 9]

[6 0 5 1 1 1 3 2]

[2 6 4 3 5 5 4 6]]

Note that there are no part marks for a partially correct network structure. Do not assume inputs have been flattened prior to being fed into the forward pass.

<h2>Task 2 – Loss</h2>

Implement a correct loss function (NNModel.lossfn). You may (and should) make calls to PyTorch here. See the comment for further information.

<h2>Task 3 – FeedForward Network</h2>

Implement a feedforward network according to the specifications in the accompanying docstring. Task 4 – Convolutional Network

Implement a convolutional network according to the specifications in the accompanying docstring.