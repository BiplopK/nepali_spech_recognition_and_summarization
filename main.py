# Importing necessary libraries
from transformers import pipeline
from pydub import AudioSegment
import numpy as np
import librosa
import time
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import soundfile as sf 
import pyloudnorm as pyln 
import numpy as np


## Functions
def load_huggingface_model(model_name):
    """
    Loads a pre-trained model from Hugging Face's Transformers library for automatic speech recognition (ASR).

    Parameters:
    -----------
    model_name : str
        The name or identifier of the pre-trained model available on Hugging Face's model hub.

    Returns:
    --------
    trained_model : Hugging Face pipeline or None
        Returns the loaded ASR model pipeline if successful. If an error occurs during model loading, 
        returns None and prints the error message.

    Raises:
    -------
    Prints an error message if the model fails to load due to any exception.

    Example:
    --------
    >>> model = load_huggingface_model("facebook/wav2vec2-large-960h")
    >>> if model:
    >>>     result = model("Audio file path or data")

    """
    try:
        print("Loading Model from Huggingface")
        # loading the model from huggingface
        trained_model = pipeline("automatic-speech-recognition",model_name)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load the model. Following error occured:\n{e}")
        return None
    return trained_model

def load_local_model(model_path, processor_path, device="cpu"):
    """
    Loads a pre-trained Wav2Vec2 model and processor from local directories for automatic speech recognition (ASR).

    Parameters:
    -----------
    model_path : str
        The file path to the pre-trained Wav2Vec2 model directory.
    
    processor_path : str
        The file path to the Wav2Vec2 processor directory, used for tokenization and feature extraction.

    device : str, optional (default="cpu")
        The device on which to load the model, either 'cpu' or 'cuda' for GPU.

    Returns:
    --------
    model : Wav2Vec2ForCTC
        The loaded Wav2Vec2 model for ASR.

    processor : Wav2Vec2Processor
        The processor corresponding to the loaded model, used for feature extraction.

    Raises:
    -------
    Prints an error message if the model or processor fails to load due to any exception.

    Example:
    --------
    >>> model, processor = load_local_model("/path/to/model", "/path/to/processor", device="cuda")
    >>> if model and processor:
    >>>     # Use model and processor for speech recognition tasks
    >>>     input_values = processor("Audio file path", return_tensors="pt").input_values
    >>>     logits = model(input_values.to("cuda")).logits

    """
    try:
        print("Loading Local Model")
        # loading the model from local directory
        model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
        processor = Wav2Vec2Processor.from_pretrained(processor_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load the model. Following error occured:\n{e}")
        return None
    return model, processor

def getFileNameAndExtension(file_path):
    """
    Description:
    -------------
    This function takes a file path as input and it will return the filename and extension for that file path
    
    Arguments:
    -------------
    file_path: Absolute or relative path for the input file
    
    Returns:
    -------------
    filename: name of the file
    extension: extension of file

    Example:
    --------
    >>> getFileNameAndExtension("/path/to/file.txt")
    ('file', 'txt')
    """
    # get the basename of the file from its absolute path
    basename = os.path.basename(file_path)
    # get the filename and file extension
    filename, ext = basename.split(".")
    return filename,ext

def segmentLargeArrayforHuggingface(inputArray,array_length,chunksize=200000):
    """
    Segments a large input array into smaller chunks based on the specified chunk size.

    Parameters:
    -----------
    inputArray : list or numpy.ndarray
        The large input array to be segmented.
    
    array_length : int
        The length of the input array. This should match `len(inputArray)` for correct segmentation.
    
    chunksize : int, optional (default=200000)
        The size of each segment. If the array length is not perfectly divisible by `chunksize`,
        the last segment may be smaller than `chunksize`.

    Returns:
    --------
    list_of_segments : list of lists or list of numpy.ndarray
        A list containing the segmented parts of the original array. Each segment is a portion of 
        the input array with a length of up to `chunksize`.

    Example:
    --------
    >>> input_array = [i for i in range(600000)]
    >>> segments = segmentLargeArrayforHuggingface(input_array, len(input_array), chunksize=200000)
    >>> len(segments)
    3  # Three segments with up to 200000 elements each.
    
    Notes:
    ------
    - This function slices the input array based on the `chunksize` and operates over the first dimension.
    - If `array_length` is smaller than `chunksize`, the entire array is returned as a single segment.
    """
    list_of_segments = []
    for i in range(0,array_length+1,chunksize):
        list_of_segments.append(inputArray[i:i+chunksize])
    return list_of_segments 

def segmentLargeArrayForLocal(inputTensor,chunksize=200000):
    """
    Segments a large input tensor into smaller chunks based on the specified chunk size.

    Parameters:
    -----------
    inputTensor : torch.Tensor or numpy.ndarray
        The input 2D tensor or array to be segmented. The function assumes that the tensor has at least 2 dimensions, 
        where the second dimension is segmented.
    
    chunksize : int, optional (default=200000)
        The size of each segment along the second dimension. If the tensor length is not perfectly divisible by the 
        chunk size, the last segment may be smaller.

    Returns:
    --------
    list_of_segments : list of torch.Tensor or numpy.ndarray
        A list containing the segmented tensors or arrays. Each segment is a portion of the original input 
        tensor along its second dimension.

    Example:
    --------
    >>> input_tensor = torch.randn(3, 600000)
    >>> segments = segmentLargeArrayForLocal(input_tensor, chunksize=200000)
    >>> len(segments)
    3  # Three segments with 200000 elements each along the second dimension.
    
    Notes:
    ------
    - This function operates along the second dimension of a 2D tensor or array.
    - If `inputTensor.shape[1]` is smaller than `chunksize`, the entire tensor is returned as a single segment.
    """
    # print(inputTensor)
    list_of_segments = []
    tensor_length = inputTensor.shape[1]
    for i in range(0,tensor_length+1,chunksize):
        list_of_segments.append(inputTensor[:,i:i+chunksize])
    return list_of_segments 

def adjust_volume(data,sr=16000,norm="peak"):
    """
    Adjusts the volume of the audio data using either peak normalization or loudness normalization.

    Parameters:
    -----------
    data : numpy.ndarray
        The input audio data, typically a 2D array where the second dimension represents the audio samples.
    
    sr : int, optional (default=16000)
        The sample rate of the audio data, used for the loudness measurement. Defaults to 16,000 Hz.
    
    norm : str, optional (default="peak")
        The type of normalization to apply:
        - "peak": Peak normalization adjusts the volume so that the loudest peak reaches -1 dB.
        - "fixed": Loudness normalization adjusts the volume to a fixed loudness level (0 dB).
        - Any other value will leave the audio unchanged.

    Returns:
    --------
    peak_normalized_audio : numpy.ndarray
        The normalized audio data after applying the specified normalization technique.

    Example:
    --------
    >>> normalized_audio = adjust_volume(audio_data, sr=16000, norm="peak")
    
    Notes:
    ------
    - Peak normalization adjusts the volume relative to the highest peak in the audio file.
    - Loudness normalization adjusts the overall perceived loudness to a fixed level (0 dB by default).
    - The `pyln.Meter` object is used to measure loudness according to ITU BS.1770 standards.
    """
    # Peak normalization of all audio to -1dB
    meter = pyln.Meter(sr) #create BS.1770 Meter
    # print(data)
    # print(np.transpose(data).shape)
    loudness = meter.integrated_loudness(np.transpose(data)) 
    # print(f'Before: {loudness} dB')
    if norm == "peak":
        # This is peak normalization which depends on the original volume of audio file
        peak_normalized_audio = pyln.normalize.peak(data,-1.0)
    elif norm=="fixed":
        # Actually this is loudness normalization to a fixed level irrespective of volume in original file
        peak_normalized_audio = pyln.normalize.loudness(data, loudness, 0)
    else:
        peak_normalized_audio = data
    loudness = meter.integrated_loudness(np.transpose(peak_normalized_audio)) 
    # print(f'After peak normalization: {loudness} dB')
    return peak_normalized_audio

def convertAudio(src_audio_path,format="mp3"):
    """
    Description:
    -------------
    This function converts audio files from m4a files to mp3 files(by default) or any other file format specified by the user
    
    Arguments:
    -------------
    src_audio_path: path of the audio file in m4a format
    
    Returns:
    -------------
    dest_audio_path: path of the audio file in desired format
    """
    filename, ext = getFileNameAndExtension(src_audio_path)
    # if the extension is already flac no need to convert
    if ext == format:
        return src_audio_path
    # create a temporary flac file as flac is supported in torchaudio
    dest_audio_path = f"temp/{filename}.{format}"  
    # using AudioSegment from pydub to convert from any audio format to flac as flac is compressed format of wav and torchaudio only supports wav and flac
    audio = AudioSegment.from_file(src_audio_path,format="m4a")
    # Export the audio to flac file
    audio.export(dest_audio_path, format=format)
    return dest_audio_path

def deleteTempAudio(dest_audio_path):
    """
    Description:
    -------------
    This function deletes the temporary converted audio files
    
    Arguments:
    -------------
    dest_audio_path: path of the audio file in mp3 format created temporarily
    
    Returns:
    -------------
    None
    """
    os.remove(dest_audio_path)

def writeOutputToFile(output, audio_file_path,translated=False):
    """
    Description:
    -------------
    This function writes the generated Nepali transcript to a file inside the output folder
    
    Arguments:
    -------------
    output: Nepali text transcript generated by ASR model
    audio_file_path: File path of the input audio
    
    Returns:
    -------------
    destination_file_path: File path of the text transcript
    """
    filename,ext = getFileNameAndExtension(audio_file_path)

    destination_file_path = f"./transcripts/{filename}.txt"
    with open(destination_file_path,"w",encoding="utf-8") as f:
        f.write(output)
    return destination_file_path

def generateTranscriptFromHuggingFaceModel(audio_input,model):
    """
    Description:
    -------------
    This function generates Nepali transcript for the Nepali speech input
    
    Arguments:
    -------------
    audio_input: Path of audio file for which transcript is generated
    model: pretrained ASR model to perform speech recognition
    
    Returns:
    -------------
    output: Nepali text transcript for the given audio input
    """
    # set mono=True as the SpeechRecognitionPipelince can only work with mono audio, Also sampling rate is set to 16k as the model was trained at this sampling rate
    speech_array, sr = librosa.load(audio_input,mono=True,sr=16000)
    array_length = speech_array.shape[0]
    # print(speech_array, array_length)
    # for longer audio, segmentation needs to be done to prevent program from consuming entire RAM which may cause error, so I am diving the entire audio to smaller segments and will process these segments
    if array_length > 250000:
        list_of_segments = segmentLargeArrayforHuggingface(speech_array,array_length, 200000)
        # print(list_of_segments)
        output = ''
        for segment in list_of_segments:
            output += model(segment)["text"]
    else:
        output = model(audio_input)["text"]
    return output


def generateTranscriptFromLocalModel(input_file,model, processor, device="cpu", do_segment=True):
    """
    Generates a transcript from an audio file using a pre-trained local speech recognition model.

    Parameters:
    -----------
    input_file : str
        The file path to the input audio file to be transcribed.
    
    model : torch.nn.Module
        The pre-trained speech recognition model (e.g., Wav2Vec2ForCTC) used for generating predictions.
    
    processor : Wav2Vec2Processor
        The processor used to prepare input data and decode model outputs, including tokenization and feature extraction.
    
    device : str, optional (default="cpu")
        The device on which to perform inference, such as "cpu" or "cuda" for GPU inference.
    
    do_segment : bool, optional (default=True)
        If True, the function will segment large audio files (greater than 10 seconds) into smaller chunks for processing.
        If False, the entire audio file will be processed in one go, irrespective of its length.

    Returns:
    --------
    output : str
        The generated transcript of the input audio file.

    Example:
    --------
    >>> transcript = generateTranscriptFromLocalModel("audio.wav", model, processor, device="cuda", do_segment=True)
    >>> print(transcript)

    Notes:
    ------
    - The function first loads the audio file and resamples it to 16,000 Hz for model compatibility.
    - Audio longer than 10 seconds is segmented into smaller chunks of 200,000 samples if `do_segment=True`.
    - The function normalizes the audio using fixed loudness normalization and converts it to a format suitable 
      for the model.
    - For each segment or the full audio, logits are obtained from the model and decoded into a transcript.

    """
    speech_array, sampling_rate = torchaudio.load(input_file)  
    speech_array = speech_array.numpy()
    speech_array = adjust_volume(speech_array,sampling_rate,norm="fixed") 
    speech_array = torch.from_numpy(speech_array) 
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)    
    resampled_array = resampler(speech_array).squeeze()
   
    if len(resampled_array.shape) == 1:
        resampled_array = resampled_array.reshape([1,resampled_array.shape[0]])
    # print(resampled_array.shape[1])
    if resampled_array.shape[1] >= 200000 and do_segment == True:
        print('The input file is longer than 10 seconds')
        list_of_segments = segmentLargeArrayForLocal(resampled_array)
        # print(list_of_segments)
        output = ''
        for segment in list_of_segments:
            if segment.size()[1] > 0:
                logits = model(segment.to(device)).logits
                # print(logits)
                pred_ids = torch.argmax(logits,dim=-1)[0]
                output += processor.decode(pred_ids)
            else:
                output += ''
    else:
        print('The input file is less than 10 seconds')
        logits = model(resampled_array.to(device)).logits
        # print(logits)
        pred_ids = torch.argmax(logits, dim = -1)[0]
        # print("Prediction:")
        output = processor.decode(pred_ids)
    
    return output

def generateTranscriptForFile(input_file_path, hf_model, local_model, local_processor, model_type="huggingface"):
    """
    Description:
    -------------
    This function generates transcript for a single audio file
    
    Arguments:
    -------------
    input_file_path: Path to the input audio file
    model: Pretrained ASR model to generate transcript
    
    Returns:
    -------------
    None
    """
    audio_extensions = ['mp3','wav','flac','m4a']
    filename, ext = getFileNameAndExtension(input_file_path)
    if ext in audio_extensions:
        print(f"{input_file_path} is a valid audio file, so proceeding to generate transcript")
        if ext == "m4a":
            print(f"{filename} is in m4a format, so converting it to mp3 format")
            input_file_path = convertAudio(input_file_path,"mp3")
        start_time = time.time()
        if model_type == "huggingface":
            output = generateTranscriptFromHuggingFaceModel(input_file_path,hf_model)
            print(f"Transcript generated in {time.time() - start_time} seconds for {filename}")
        elif model_type == "local":
            output = generateTranscriptFromLocalModel(input_file_path,local_model,local_processor)
            print(f"Transcript generated in {time.time() - start_time} seconds for {filename}")
        if ext == "m4a":
            print(f"Deleting temporarily created mp3 file")
            deleteTempAudio(input_file_path)
        destination_file_path = writeOutputToFile(output,input_file_path)
        print(f"Transcript for {filename} is written at {destination_file_path}")
    else:
        print(f"{input_file_path} is not a valid audio file, please enter a valid audio file with extension mp3, wav or flac")

def generateTranscriptForFileUsingHF(input_file_path, hf_model):
    """
    Description:
    -------------
    This function generates transcript for a single audio file
    
    Arguments:
    -------------
    input_file_path: Path to the input audio file
    model: Pretrained ASR model to generate transcript
    
    Returns:
    -------------
    None
    """
    audio_extensions = ['mp3','wav','flac','m4a']
    filename, ext = getFileNameAndExtension(input_file_path)
    if ext in audio_extensions:
        print(f"{input_file_path} is a valid audio file, so proceeding to generate transcript")
        if ext == "m4a":
            print(f"{filename} is in m4a format, so converting it to mp3 format")
            input_file_path = convertAudio(input_file_path,"mp3")
        start_time = time.time()
        output = generateTranscriptFromHuggingFaceModel(input_file_path,hf_model)
        print(f"Transcript generated in {time.time() - start_time} seconds for {filename}")
        if ext == "m4a":
            print(f"Deleting temporarily created mp3 file")
            deleteTempAudio(input_file_path)
        destination_file_path = writeOutputToFile(output,input_file_path)
        print(f"Transcript for {filename} is written at {destination_file_path}")
        return output
    else:
        print(f"{input_file_path} is not a valid audio file, please enter a valid audio file with extension mp3, wav or flac")
        return None

# # t3 = time.time()
# # generateTranscriptForFile('./input/anushasan.m4a',hf_model, l_model, l_processor, model_type="local")
# # t4 = time.time()
# # print(f"Time for local model: {t4-t3} seconds")

# hf_model = load_huggingface_model("anish-shilpakar/wav2vec2-nepali")
# l_model, l_processor =  load_local_model("./model", "./processor")

# t1 = time.time()
# generateTranscriptForFile('./input/anushasan.m4a',hf_model, l_model, l_processor, model_type="huggingface")
# t2 = time.time()
# print(f"Time for huggingface model: {t2-t1} seconds")