# Gesture Recognition – Deep Learning

We have developed a smart-TV feature that recognizes **five different gestures**, each mapped to a specific command:

- **Thumbs up**: Increase the volume  
- **Thumbs down**: Decrease the volume  
- **Left swipe**: Jump backward 10 seconds  
- **Right swipe**: Jump forward 10 seconds  
- **Stop**: Pause the movie  

---

## Training Data

To train our models, we collected a few hundred videos categorized into these five classes. Each video:
- Is **2–3 seconds** long.  
- Is **divided into a sequence of 30 frames (images)**.  
- Was recorded by different people, performing one of the five gestures in front of a webcam.

---

## Training Environment – Google Colab with A100 GPU

For rapid experimentation and handling large video datasets, we used **Google Colab** with an **NVIDIA A100 GPU**. The A100 provides:

- **High Performance**: Faster matrix computations and model training.  
- **Large Memory**: Supports larger batch sizes and higher-resolution frames.  
- **Scalability**: Ideal for training both 3D Convolutional Networks (Conv3D) and CNN + RNN architectures.

This setup was essential to achieving high accuracy and efficient model tuning on our dataset.

---

## Training Models

We experimented with **two common architectures** for video analysis:

### 1. Conv3D (3D Convolutional Network)
- Uses 3D filters to learn **spatial (x, y) and temporal (z)** information from multiple frames at once.  
- More direct approach, treating the entire video volume as a 3D tensor.

### 2. CNN + RNN (Convolution + Recurrent Neural Network)
- A **2D CNN (Conv2D)** extracts features from each frame.  
- Those features are fed into an **RNN** (e.g., GRU or LSTM), which learns temporal patterns.  
- The RNN output is passed to a softmax layer to classify gestures.  

---

## Results of Our Experiments

We ran several experiments to evaluate the **validation accuracy** of different hyperparameter settings—varying:  
- **img_index** (number of frames used per video)  
- **Dim** (image dimension: y, z)  
- **Batch size**  
- **Number of layers**  
- **Learning rate (LR)**  
- **Dropout rate**  

The table below summarizes each experiment:

| **Experiment #** | **Model**     | **Hyperparameters**                                                          | **Validation Accuracy** | **Decision / Explanation**                                                                                                                                                                                                                     |
|------------------|---------------|------------------------------------------------------------------------------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1                | **Conv3D**    | Batch size = 32, <br> img_index = 13, <br> Dim = 80×80, <br> Epoch = 20       | 0.71                    | - Best performing model <br> - Accuracy plateaus around epoch 12 <br> - Large model (many parameters) and long training time, potentially leading to overfitting                                                                                  |
| 2                | **Conv2D+GRU**| Batch size = 5, <br> img_index = 13, <br> Dim = 80×80, <br> Epoch = 20        | 0.25                    | - Very unstable model <br> - Inconsistent learning <br> - Much worse than Model 1                                                                                                                                                                |
| 3                | **Conv3D**    | Batch size = 5, <br> img_index = 13, <br> Dim = 120×120, <br> Epoch = 20      | 0.63                    | - Tweaked original Conv3D <br> - Larger image dimension, fewer layers <br> - Better than Model 2 but not as good as Model 1 <br> - Trains faster than Model 1 (fewer parameters)                                                                |
| 4                | **Conv3D**    | Batch size = 5, <br> img_index = 30, <br> Dim = 120×120, <br> Epoch = 20      | 0.56                    | - Performance dropped <br> - Model is more complex (due to more frames) and takes longer to train <br> - Overall accuracy is lower                                                                                                                                  |

---

### Conclusion

- **Model 1 (Conv3D)**: Highest accuracy (0.71) but **longer training time** and potential overfitting due to many parameters.  
- **Model 3 (Conv3D)**: Fewer parameters and **faster training** but slightly lower accuracy (0.63). Depending on acceptable accuracy vs. training time and resource constraints, Model 3 may be sufficient.

---

