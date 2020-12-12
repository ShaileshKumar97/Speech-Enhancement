# Speech-enhancement
---

## Introduction
**This project aims at building a speech enhancement module to minimize environmental noise.**

Audios can be represented in different ways as raw time series to time-frequency decompositions.
Among time-frequency decompositions, Spectrograms is a useful representation for audio processing. They consist in 2D images representing sequences of Short Time Fourier Transform (STFT) with time and frequency as axes, and brightness representing the strength of a frequency component at each time frame. In such they appear a natural domain to apply the CNNS architectures for images directly to sound. Between magnitude and phase spectrograms, magnitude spectrograms contain most the structure of the signal. Phase spectrograms appear to show only little temporal and spectral regularities.

In this project, I will use magnitude spectrograms as a representation of sound (image below) in order to predict the noise model to be subtracted to a noisy voice spectrogram.

<img src="https://github.com/vbelz/Speech-enhancement/raw/master/img/sound_to_spectrogram.png" alt="sound representation" title="sound representation" />

The project has three pipeline:  `Dataset preparation pipeline`, `Training pipeline`, `Prediction pipeline`.

## Preparing Dataset pipeline

To create the datasets for training, I gathered english speech clean voices  and environmental noises from different sources.

The clean voices were mainly gathered from [LibriSpeech](http://www.openslr.org/12/): an ASR corpus based on public domain audio books.
The environmental noises were gathered from [ESC-50 dataset](https://github.com/karoldvl/ESC-50).  

 For this project, I only used 10 classes of environmental noise: 
 **Noise**    **Code**
 
***insects - 7***
***footsteps - 25***
***brushing_teeth - 27***
***snoring - 28***
***vaccum_cleaner - 36***
***clock_alarm - 37***
***clock_tick - 38***
***church_bells - 46***
***fireworks - 48***
***hand_saw - 49***

To create the datasets for training/validation/testing, audios were sampled at 8kHz and I extracted windows slighly above 1 second. I performed some data augmentation for the environmental noises (taking the windows at different times creates different noise windows). Noises have been blended to clean voices  with a randomization of the noise level (between 20% and 80%). At the end, training data consisted of around 10h of noisy voice & clean voice,
and validation data of 1h of sound.

I prepared dataset by by creating data/Train and data/Test folders in a location separate from code folder. Then I create the following structure as in the image below:

<img src="https://github.com/vbelz/Speech-enhancement/raw/master/img/structure_folder.png" alt="data folder structure" title="data folder structure" />

I moved my noise audio files into `noise_dir` directory and clean voice files into `voice_dir`.

I set nb_samples=50.

Then run `create_data()` function. This will randomly blend some clean voices from `voice_dir` with some noises from `noise_dir` and save the spectrograms of noisy voices, noises and clean voices to disk as well as complex phases, time series and sounds. It takes the inputs parameters defined in  `notebook`. Parameters for STFT, frame length, hop_length can be modified in `notebook`, but with the default parameters each window will be converted into spectrogram matrix of size 128 x 128.

Datasets to be used for training will be magnitude spectrograms of noisy voices and magnitude spectrograms of clean voices.

`create_data()` takes around 3 hours for 5000 samples of clean-voice and 1900 samples of noise to process while it takes 46 minutes for 2600 samples of clean-voice and 300 samples of noise. 


## Training Pipeline

The model used for the training is a U-Net, a Deep Convolutional Autoencoder with symmetric skip connections. [U-Net](https://arxiv.org/abs/1505.04597) was initially developed for Bio Medical Image Segmentation. Here the U-Net has been adapted to denoise spectrograms.

As input to the network, the magnitude spectrograms of the noisy voices. As output the Noise to model (noisy voice magnitude spectrogram - clean voice magnitude spectrogram). Both input and output matrix are scaled with a global scaling to be mapped into a distribution between -1 and 1.

<img src="https://github.com/vbelz/Speech-enhancement/raw/master/img/Unet_noisyvoice_to_noisemodel.png" alt="Unet training" title="Unet training" />

The encoder is made of 10 convolutional layers (with LeakyReLU, maxpooling and dropout). The decoder is a symmetric expanding path with skip connections. The last activation layer is a hyperbolic tangent (tanh) to have an output distribution between -1 and 1. For training from scratch the initial random weights where set with `He normal` [initializer](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal).

Model is compiled with Adam optimizer and the loss function used is the Huber loss.

I tried `training_from_scratch = True` but it takes time and the prediction from model is not such good so I used `training_from_scratch = False` by taking pretrained weights from [this repo](https://github.com/vbelz/Speech-enhancement/blob/master/weights/model_unet.h5).

The number of epochs and the batch size for training are specified by `epochs` and `batch_size`. Best weights are automatically saved during training as `model_unet.h5`.

At the end, I obtained a training loss of 0.0048 and a validation loss of 0.00579. Below a loss graph made in one of the trainings.

<img src="https://github.com/ShaileshKumar97/Speech-Enhancement/blob/main/notebook/loss_training.png?raw=true" alt="loss training" title="loss training" />

## Prediction pipeline

For prediction, the noisy voice audios are converted into numpy time series of windows slightly above 1 second. Each time serie is converted into a magnitude spectrogram and a phase spectrogram via STFT transforms. Noisy voice spectrograms are passed into the U-Net network that will predict the noise model for each window (graph below). Prediction time for one window once converted to magnitude spectrogram is around 80 ms using classical CPU.

<img src="https://github.com/vbelz/Speech-enhancement/raw/master/img/flow_prediction.png" alt="flow prediction part 1" title="flow prediction part 1" />

Then the model is subtracted from the noisy voice spectrogram (here I apply a direct subtraction as it was sufficient for my task, we could imagine to train a second network to adapt the noise model, or applying a matching filter such as performed in signal processing). The "denoised" magnitude spectrogram is combined with the initial phase as input for the inverse Short Time Fourier Transform (ISTFT). Our denoised time serie can be then converted to audio (graph below).

<img src="https://github.com/vbelz/Speech-enhancement/raw/master/img/flow_prediction_part2.png" alt="flow prediction part 2" title="flow prediction part 2" />

Here is the prediction from the model:

> Example:

[Input example mix](https://github.com/ShaileshKumar97/Speech-Enhancement/blob/main/demo/input/noisy_voice_long_t2.wav)

[Predicted output example mix](https://github.com/ShaileshKumar97/Speech-Enhancement/blob/main/demo/prediction/denoise_t2.wav)


## References

>Vinsent-belz **Speech Enhancement**.
>
>[https://github.com/vbelz/Speech-enhancement]

>Jansson, Andreas, Eric J. Humphrey, Nicola Montecchio, Rachel M. Bittner, Aparna Kumar and Tillman Weyde.**Singing Voice Separation with Deep U-Net Convolutional Networks.** *ISMIR* (2017).
>
>[https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf]

>Grais, Emad M. and Plumbley, Mark D., **Single Channel Audio Source Separation using Convolutional Denoising Autoencoders** (2017).
>
>[https://arxiv.org/abs/1703.08019]

>Ronneberger O., Fischer P., Brox T. (2015) **U-Net: Convolutional Networks for Biomedical Image Segmentation**. In: Navab N., Hornegger J., Wells W., Frangi A. (eds) *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015*. MICCAI 2015. Lecture Notes in Computer Science, vol 9351. Springer, Cham
>
>[https://arxiv.org/abs/1505.04597]

> K. J. Piczak. **ESC: Dataset for Environmental Sound Classification**. *Proceedings of the 23rd Annual ACM Conference on Multimedia*, Brisbane, Australia, 2015.
>
> [DOI: http://dx.doi.org/10.1145/2733373.2806390]
