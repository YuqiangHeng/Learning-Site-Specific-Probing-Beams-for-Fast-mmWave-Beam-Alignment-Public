# Learning-Site-Specific-Probing-Beams-for-Fast-mmWave-Beam-Alignment
This repo implements the joint probing codebook and beam selector architecture in the paper "learning site-specific probing beams for fast mmWave beam alignment".
*Heng, Yuqiang, Jianhua Mo, and Jeffrey G. Andrews. "Learning Site-Specific Probing Beams for Fast mmWave Beam Alignment." arXiv preprint arXiv:2107.13121 (2021).
*https://arxiv.org/abs/2107.13121
To run experiments using the DeepMIMO scenarios, please generate the datasets according to the specifications described in the paper, save the real and imaginary parts of the channel vectors to "h_real.npy"& "h_imag.npy" and place them in the Dataset folder under the appropriate sub-folders. 
### Model Training Scripts
*Model_Training.py
*Model_Training_vs_Channel_Estimation_Error.py
*Model_Training_vs_Measurement_Noise.py
*Model_Training_vs_Probing_Codebook_Quantization.py
### Results and Plotting Scripts
*Plot_Results_Acc_SNR_vs_Baselines.py
*Plot_Results_Acc_vs_Channel_Estimation_Error.py
*Plot_Results_Acc_vs_Measurement_Noise.py
*Plot_Results_Acc_vs_Probing_Codebook_Classifier_Architecture.py
*Plot_Results_Acc_vs_Probing_Codebook_Quantization.py
*Plot_Results_Probing_Codebook_t-SNE.py
*Plot_Results_Silhouette_Score.py
### Utility Functions and Classes
*beam_utils.py
*ComplexLayers_Torch.py
