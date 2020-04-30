# NeuralNetworks
This is the repository for Neural Networks project called Speech Emotion Classification Using Attention-Based LSTM

## Collaborators 
Nijat Mursali 

Yunus Emre Darici

## Introduction

Automatic speech emotion recognition has been a research hotspot in the field of human–computer interaction over the past decade. 
However, due to the lack of research on the inherent temporal relationship of the speech waveform, the current recognition accuracy needs improvement.
To make full use of the difference of emotional saturation between time frames, a novel method is proposed for speech recognition 
using frame-level speech features combined with attention-based long short-term memory (LSTM) recurrent neural networks. 
Frame-level speech features were extracted from waveform to replace traditional statistical features, which could preserve the timing relations in the original speech through the sequence of frames. 
To distinguish emotional saturation in different frames, two improvement strategies are proposed for LSTM based on the attention mechanism: 
first, the algorithm reduces the computational complexity by modifying the forgetting gate of traditional LSTM without sacrificing performance and second, in the final output of the LSTM, 
an attention mechanism is applied to both the time and the feature dimension to obtain the information related to the task, rather than using the output from the 
last iteration of the traditional algorithm. Extensive experiments on the CASIA, eNTERFACE, and GEMEP emotion corpora demonstrate 
that the performance of the proposed approach is able to outperform the state-of-the-art algorithms reported to date.

## Conclusion
Developed the algorithm and presented to prof. 
