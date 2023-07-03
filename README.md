# ESTT: Federated Learning for Autonomous Driving: Enhancing Style Transfer Techniques for FFreeDA
### Machine Learning and Deep Learning 2023, Politecnico di Torino, Summer Project 2B

## Abstract
Semantic Segmentation (SS) is a task fundamental to enabling the diffusion of self-driving vehicles, providing them with a tool to precisely understand their surroundings by assigning a class to each pixel of an image. However, in this application it would rely on sensitive data gathered from users’ vehicles, raising substantial privacy concerns. In response to this challenge, in this paper we explore the application of Federated Learning (FL). However, often FL is implemented under the unrealistic assumption of having labeled data at client side. Thus, we also explore the recently introduced task of FFreeDA, in which clients’ data is unlabeled and the server accesses a source-labeled dataset for pre-training. Moreover, we expand on existing style-transfer techniques by introducing two novel approaches to perform Fourier Domain Adaptation (FDA). In one, we introduce a clustering scheme using fuzzy logic at inference time. In the other, we try to mimic the presence of unseen sub-styles in the target dataset, aiming to improve the capability of generalization of the model. Both seem to yield some promising results, showing improvements upon the existing methods.

## Code structure

The mains for the various tasks can be found in:
- main (Task 2 and Task 4)
- model_t1 (Task 1)
- model_t3 (Task 3)
- model_t5.1 (Task 5 - extension 1)
- model_t5.1 (Task 5 - extension 2)

Various experiments and plots can be found in the folders Experiments and Results, respectively
