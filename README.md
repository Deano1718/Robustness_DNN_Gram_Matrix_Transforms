# Robustness_DNN_Gram_Matrix_Transforms

Code implementation of N. Dean and D. Sarkar, "DNN Feature Map Gram Matrices for Mitigating White-Box Attacks on Image Classification," 2023 International Conference on Intelligent Computing, Communication, Networking and Services (ICCNS), Valencia, Spain, 2023, pp. 111-118, doi: 10.1109/ICCNS58795.2023.10193579.

Inspired by image stylization, we provide a pre-processing defense that randomly draws K clean examples, one from each class, from the training set and imparts their Gram matrix values to K copies of the input images (K being the number of classes).  While pre-processing defenses fell out of favor after Athalye, A., Carlini, N. &amp; Wagner, D.. (2018). Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples. <i>Proceedings of the 35th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 80:274-283, we find that producing K different transformations can enhance the accuracy of adversarially trained networks.  We provide evaluations using both naive- and full-knowledge white-box attackers.

