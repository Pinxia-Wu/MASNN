# MASNN: Multi-scale adaptive subspace neural networks for solving partial differential equations with high accuracy

Pinxia Wu, Zhiqiang Sheng

Deep neural networks (DNNs) enhanced with Fourier feature mappings outperform traditional DNNs in
fitting multi-scale and high-frequency functions. However, the initialization parameters of the feature mappings for
such networks often heavily depend on the frequency characteristics of the target functions, which typically results
in low numerical accuracy and thereby restricts their broader applicability. To address multi-scale challenges, this
paper proposes a multi-scale adaptive subspace neural network (MASNN). Specifically, this algorithm employs a series
of DNN frameworks to analyze the spectrum of previous network solution to obtain the initialization parameters of
Fourier feature embeddings in the current neural network, and can accurately capture the frequency information of the
exact solution after several adaptive adjustments without prior frequency information. Additionally, a subspace layer is
incorporated into the network architecture to improve the accuracy of prediction, which obtains the basis functions of
the solution space through network training and then solves the algebraic system by enforcing the governing equation
along with initial/boundary conditions to acquire the high-resolution solution.
