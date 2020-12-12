#!/usr/bin/env python
# coding: utf-8

# In[ ]:


noise_dir = '/content/Train/noise'                        #folders where to find noise audios to prepare training dataset
voice_dir = '/content/Train/clean_voice'                  #folders where to find clean voice audio to prepare training dataset
path_save_spectrogram = '/content/Train/spectrogram/'     #folders where to save spectrograms for training / QC
path_save_time_serie = '/content/Train/time_serie/'       #folders where to save time series for training / QC
path_save_sound = '/content/Train/sound/'                 #folders where to save sounds for training / QC
nb_samples = 50                                           #How much frame to create in create_data()
training_from_scratch = False                             #Training from scratch or pre-trained weights
weights_path = '/content/Train/model/'                    #folder of saved weights
epochs = 50                                               #No of epochs for training
batch_size = 30                                           #Batch size for training
name_model = 'model_unet'                                 #Name of saved model to read
audio_dir_prediction = '/content/Test/noisy_voice'        #directory where prediction() function read noisy sound for denoise
dir_save_prediction = '/content/Test/save_predictions/'   #directory to save the denoise sound
audio_input_prediction = ['noisy_voice_long_t2.wav']      #Noisy sound file to denoise
audio_output_prediction = 'denoise_t2.wav'                #File name of sound output of denoise prediction
sample_rate = 8000                                        #Sample rate chosen to read audio
min_duration = 1.0                                        #Minimum duration of audio files to consider
frame_length = 8064                                       #Training data will be frame of slightly above 1 second
hop_length_frame = 8064                                   #hop length for clean voice files separation (no overlap
hop_length_frame_noise = 5000                             #hop length for noise files to blend (noise is splitted into several windows)
n_fft = 255                                               #Choosing n_fft to have squared spectrograms
hop_length_fft = 63                                       #Choosing hop_length_fft to have squared spectrograms