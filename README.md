# Learning-Site-Specific-Probing-Beams-for-Fast-mmWave-Beam-Alignment

This repo implements the joint probing codebook and beam selector architecture in the paper "learning site-specific probing beams for fast mmWave beam alignment".
* Heng, Yuqiang, Jianhua Mo, and Jeffrey G. Andrews. "Learning Site-Specific Probing Beams for Fast mmWave Beam Alignment." arXiv preprint arXiv:2107.13121 (2021).
* https://arxiv.org/abs/2107.13121

To run experiments using the DeepMIMO scenarios, please generate the datasets according to the specifications described in the paper, save the real and imaginary parts of the channel vectors to "h_real.npy"& "h_imag.npy" and place them in the Dataset folder under the appropriate sub-folders. 

### Model Training Scripts

* Model_Training.py
  * Trains and saves models with MLP beam selection function that takes the measured power of learnable probing codebooks, even-spaced DFT probing codebooks and AMCF wide-beam probing codebooks, CNN beam selection function that takes the complex received signals of evenly-spaced DFT probing codebooks. 
* Model_Training_vs_Channel_Estimation_Error.py
  * Trains and saves models (learnable probing codebook + MLP classifier) with increasing channel estimation error 
* Model_Training_vs_Measurement_Noise.py
  * Trains and saves models (learnable probing codebook + MLP classifier) under increasing probing codebook measurement noise
* Model_Training_vs_Probing_Codebook_Quantization.py
  * Trains and saves models under different phase-shifter resolutions for the learned probing codebook.
### Results and Plotting Scripts

* Plot_Results_Acc_SNR_vs_Baselines.py
  * Plot the beam alignment accuracy and average SNR of the proposed method and compare to baselines 
* Plot_Results_Acc_vs_Channel_Estimation_Error.py
  * Plot the beam alignment accuracy of the propsoed method with increasing channel estimation error 
* Plot_Results_Acc_vs_Measurement_Noise.py
  * Plot the beam alignment accuracy of the proposed method with different probing codebook measurement noise
* Plot_Results_Acc_vs_Probing_Codebook_Classifier_Architecture.py
  * Plot the beam alignment accuracy with different architectures (differnet probing codebooks, different classifiers)
* Plot_Results_Acc_vs_Probing_Codebook_Quantization.py
  * Plot the beam alignment accuracy of the proposed method vs. phase-shifter resolution 
* Plot_Results_Probing_Codebook_t-SNE.py
  * Visualize the probing codebook embedding of the channel vectors in 2D (clustering). 
* Plot_Results_Silhouette_Score.py
  * Compute the silhouette scores of the probing codebook embedding of the channel vectors (clustering). 

### Utility Functions and Classes

* beam_utils.py
  * beamforming-related utility functions and classes 
* ComplexLayers_Torch.py
  * Pytorch implementation of deep complex network linear layer under constant modules constraint
  * Pytorch implementation of joint probing codebook and beam selection function neural network model
