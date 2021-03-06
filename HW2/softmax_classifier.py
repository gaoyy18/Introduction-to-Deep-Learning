import numpy as np

def softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    # TODO: Put your code here

    N, C = np.shape(label)

    alpha = np.exp(np.dot(input, W))  #denominator of h_k(x)
    hk = np.true_divide(alpha, np.tile(np.sum(alpha, axis=1), (C, 1)).T)  #h_k(x), result after softmax, shape(N, C)

    loss = -1/N * np.sum(np.multiply(label, np.log(hk))) + 0.5 * lamda * np.square(np.linalg.norm(W))  #loss, a float number

    gradient = 1/N * np.dot(input.T, (hk - label)) + lamda * W  #gradient, shape(D, C)

    prediction = np.argmax(hk, axis=1)  #prediction, shape(N,1)


    ############################################################################

    return loss, gradient, prediction
