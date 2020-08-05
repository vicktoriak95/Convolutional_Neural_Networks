r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Increasing k does not lead to improved generalization. For example, consider a case in which k is equal to the number of samples in the training case. In such a case, the prediction will always be the same - the majority label among all of the samples. This, of course, is not a good generalization. For big k values, we might consider irrelevant samples, outside of the relevant cluster. 
With that said, a minimal k (k = 1) is very influenced by noise. Therefore, we would like k to be small, but not too small. Increasing k might lead to improved generalization, as long as k is small enough, such that the k nearest neighbors are usually in the relevant cluster. This way, we will consider the majority among k samples which are probably relevant. 
In the case above, we have seen that the best k value is 3. 


"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


The two hyperparameters which are chosen for the SVM loss are lambda, Delta. The value of lambda affects the magnitude of the weights. 
We can choose Delta arbitrarily, because every choice of Delta can be balanced by lambda accordingly, in order to receive the desired outcome.
Meaning, if we choose a small value of Delta, we would like the differences between the correct and incorrect predictions to be smaller. Thus, we will use a smaller value for lambda, which will make the wights as small as needed.
If we choose a large value of Delta, we would like the differences between the predictions to be big enough. Therefore, we will use lambda which will increase the weights as needed. 
In fact, for any choice of \Delta we can find a value for lambda such that the weights will be suited for the safety margin Delta.


"""

part3_q2 = r"""
**Your answer:**


1.The model seeks a plane in the space which divides the classes well. Therefore, for non-linear data, we cannot find a good linear classifier, and the prediction will not be very accurate.

2.KNN compares the tested value to the k nearest values on the train. However, the linear classifier evaluates a prediction using the calculated weights which are based on the train set. KNN algorithm uses less information from the train set. 


"""

part3_q3 = r"""
**Your answer:**


1.The learning rate is good. The loss graph descends rapidly yet smoothly. If the LR was too high, we would expect to see spikes in the loss graph, which indicates that in each step we advance too much in the direction of the gradient. Therefore, the result does not necessarily improve. If the LR was too low, we would see a mild descent. 


2. The model somewhat overfits. We can recognize that based on the fact that the train accuracy is bigger than the valid accuracy. Meaning, the accuracy on the train was improved greatly, yet the accuracy on the valid set is not as good. This, we conclude that the model overfits the training set.


"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
The ideal pattern is a narrow scatter within the sleeve of average. 
It seems that the more the data concentrated inside the sleeves, the fit of the model is more accurate.
We can see that the model that fits the data the best (both training and test) is the cross-validation trained model.
It is clear from the residual plot that it fits the data better than the model based on top-5 features. 


"""

part4_q2 = r"""
**Your answer:**
1. Using logspace lets us sample a wide range of values for lambda, with small number of samples. 
Using logspace we can check lambda values in different order of magnitude. 

2. the model was fitted to data 3(num of degrees we tested) * 20(num of lambdas we tested) * 3(num of folds) = 180 times.


"""

# ==============
