It is my intellectual property, if you want to use it you are bound to let me know the purpose (simoneambrogio80@gmail.com).


# Qiskit-Perceptron-MNIST-DB-set
It binary classifies handwritten digits, no training.


# REQUIREMENTS
I used IMB Qiskit Library for Python 3.6.
HEre you also need imageio and elasticdeform libraries.


# PURPOSE
In this repo you can find the MNIST dataframe problem. A harder problem, but faced by the same 3-qubits circuit as in Qiskit-Perceptron-iris-DB-set.
I commented as much I can and the aim of the algo is to classify each sample inputed (X[row]) in an unsupervised binary fashion.


# STRUCTURE
Library 'DB_MNIST_checkpunctual.py' loads the files, preprocesses the data and creates/distributes the classes,
Library 'Generative_Layer_for_DB_MNIST_FWD.py' serves for creating unknown handwritten images (via imageio and elasticdeform) so to test the error when 'Qperceptron_mnistcomm.py' tells you 'same class' or 'different class' after measurement over 3 qubits. The manipulation of the qubits is my personal variation of QFT algos and the algo is the same as Qiskit-Perceptron-iris-DB-set one.


# PARAMETERS
Only relevant parameters are:

- what_number_label (you can type in a digit between 0 and 9)

- the_other_class_num_array (array of digits against which you want to compare the digit above)

- divisorpower [NO NEED TO CHANGE IT] (it is a function of the number of columns [features] in the dataframe... ~0.13 x the number of columns... no need to play with it in this stage)

- REAL_DEVICE (True if you want to run on ibm real quantum machine... I use a token.txt after registration in https://quantum-computing.ibm.com/).


# PRINTOUT
In the end it prints the accuracy. I used 2 models ('straight' that leverages the function qft, and 'reverse' that leverages the function qft_rev).
Note that the names 'straight', 'reverse' to the models are for identification purpose only. There is no straight and reverse indeed.

Case 1: CLASSIFIER. In the case the_other_class_num_array is 1-element array (e.g.: [4]). In this case what_number_label (e.g.: 4) will be checked against the_other_class_num_array. They are totally different sets ('Generative_Layer_for_DB_MNIST_FWD.py' created new 4s as described above),
you can ignore BINARY GUESS: ... and SOLUTIONS: ..., and you can focus on ERROR SCORES: [[50.] [48.]] only. In that example, the algorithm failed to classify the 2 sets, being the same digit. But if the digits differs you may get a high deviation from the 50, such as ERROR SCORES: [[82.] [88.]]. In this latter example, the algorithm classifies the 2 sets as different. 

Case 2: RECOGNIZER. In the case the_other_class_num_array is 3-elements array or larger than 3-elements (e.g.: [3,4,7]). In this case what_number_label (e.g.: 3) will be checked against all the digits in the_other_class_num_array, one by one. You will get a modified-entropy table and - most of the times - a decision,
in other words the algo uses the modified-entropy in order to dischard the digits that are different (ENTROPY[kk] >= 2.05) and to make educated guesses on the digit of the X dataframe (e.g.: 'BINARY GUESS: 3' because the digit with the least cumulated modified-entropy is the winner = the most similar).



