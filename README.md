# live-speaker-id

![](https://blog.kakaocdn.net/dn/bhFImS/btsIWeuTKuS/S7dyNd4GB7VSMrqUcnqeBk/img.png)

This system is a real-time speaker identification and transcription tool that captures audio input, identifies different speakers, and transcribes their speech into text. By recording samples of the user's voice during an enrollment phase to create a unique voice embedding, it can distinguish between the user ("Me") and other speakers.

---

# Real-Time Speaker Identification and Transcription with Python  

![](https://appliedmachinelearning.wordpress.com/wp-content/uploads/2017/11/speakerid.jpg)
   
Speaker identification and transcription are critical components in many modern applications, such as voice-controlled devices, teleconferencing tools, and surveillance systems. This post presents a comprehensive Python script that captures audio in real-time, identifies speakers, and transcribes speech using advanced models and libraries.  
   
The script performs the following tasks:  
   
- **Enrollment**: Records samples of the primary user's voice to create an averaged voice embedding.  
- **Real-time Processing**: Captures audio input, detects speech segments, identifies the speaker, and transcribes the speech.  
   
---  
   
## Overview of the Implementation  
   
The script leverages several libraries and models:  
   
- **[Vosk](https://alphacephei.com/vosk/)**: An offline speech recognition toolkit.  
- **[WebRTC Voice Activity Detector (VAD)](https://github.com/wiseman/py-webrtcvad)**: Detects speech segments in audio.  
- **[SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)**: Generates embeddings for speaker identification.  
- **[SoundDevice](https://python-sounddevice.readthedocs.io/en/0.4.6/)**: Captures audio from the microphone.  
   
---  
   
## Key Components  
   
### Voice Activity Detection (VAD)  
   
VAD is essential for determining when someone is speaking. The script uses WebRTC's VAD to process audio frames and detect voiced segments.  
   
```python  
self.vad = webrtcvad.Vad(self.VAD_AGGRESSIVENESS)  
```  
   
- **Aggressiveness Levels**: The `VAD_AGGRESSIVENESS` parameter ranges from 0 to 3, with higher values being more aggressive in filtering out non-speech.  
   
### Speech Recognition with Vosk  
   
Vosk provides offline speech recognition capabilities.  
   
```python  
self.model = Model(self.MODEL_NAME)  
```  
   
- **Model Setup**: The script expects a Vosk model directory specified by `MODEL_NAME`, such as `'models/vosk-model-small-en-us-0.15'`.  
   
### Speaker Embeddings with SpeechBrain  

SpeechBrain's ECAPA-TDNN model is a powerful tool for generating embeddings that encapsulate a speaker's unique voice characteristics. By leveraging advanced neural network architectures and attention mechanisms, it achieves high performance in speaker recognition tasks while remaining efficient and adaptable.

SpeechBrain's ECAPA-TDNN model generates embeddings that represent a speaker's voice characteristics.  
   
```python  
self.encoder = EncoderClassifier.from_hparams(  
    source="speechbrain/spkrec-ecapa-voxceleb",  
    ...  
)  
```  
   
- **Embeddings**: Voice embeddings are high-dimensional vectors used to represent and compare voices.  
   
---  

### Impact of Increasing Number of Speakers on Identification Accuracy  

![](https://picovoice.ai/static/09e129c2edde6d3923213efc9db30a17/4a02d/how_speaker_identification_works.png)
   
As the number of speakers identified by the system grows, several factors affect the accuracy and reliability of speaker identification.  
   
#### Variability and Similarity Scores  
   
- **Increased Overlap in Voice Characteristics**: With more speakers, there's a higher chance that some will have similar voice characteristics. Factors like similar pitch, tone, accent, or speaking style can cause voice embeddings to be closer in the high-dimensional space.  
   
- **Higher Similarity Scores Between Different Speakers**: As a result of similar voice traits, the cosine similarity scores between different speakers' embeddings may increase. This means that different speakers might produce embeddings that are more similar to each other, leading to higher similarity scores.  
   
- **Challenges in Distinguishing Speakers**:  
  - **Fixed Threshold Limitations**: The fixed similarity threshold (`self.THRESHOLD`) used to determine if a speaker matches a stored embedding may become less effective. Similarity scores may cluster together, making it difficult to separate speakers using a single threshold value.  
  - **Potential for Misidentification**: Higher similarity scores between different speakers increase the risk of false positives (incorrectly identifying a speaker) and false negatives (failing to recognize a speaker).  
   
#### Computational Considerations  
   
- **More Comparisons per Segment**:  
  - **Increased Computational Load**: The system must compare the current speaker's embedding against all stored embeddings, including the primary user and all previously identified speakers. As the number of speakers grows, the number of comparisons increases linearly.  
  - **Impact on Processing Time**: More comparisons can lead to longer processing times for each speech segment, potentially affecting the system's real-time performance.  
   
- **Impact on Real-Time Performance**:  
  - **Latency Issues**: The increased computational demand may introduce latency, causing delays in speaker identification and transcription outputs.  
  - **Resource Utilization**: On systems with limited processing power, the additional computational load can strain resources, leading to decreased overall system performance.  
   
#### Practical Implications  
   
- **Identification Accuracy May Decrease**: With the addition of more speakers, the system may struggle to maintain high accuracy due to the increased likelihood of similar embeddings among different speakers.  
   
- **System Scalability**:  
  - **Effective in Controlled Environments**: The system works best in settings with a limited and known set of speakers, such as small meetings or controlled group discussions.  
  - **Challenges in Open Environments**: In public or dynamic environments with many speakers, the system may face difficulties in accurately distinguishing and tracking speakers.  
   
#### Mitigation Strategies  
   
- **Dynamic Threshold Adjustment**:  
  - **Adaptive Thresholds**: Instead of a fixed threshold, implement adaptive thresholds based on the distribution of similarity scores. This approach can help accommodate variations in similarities due to additional speakers.  
   
- **Limit the Number of Tracked Speakers**:  
  - **Focus on Key Speakers**: Configure the system to track and identify a finite set of important or frequently occurring speakers, which can reduce computational load and improve accuracy.  
   
- **Enhanced Algorithms**:  
  - **Clustering Techniques**: Use clustering algorithms to group similar embeddings, reducing the number of comparisons by only comparing within relevant clusters.  
  - **Machine Learning Classifiers**: Integrate more sophisticated machine learning models trained to distinguish between specific speakers, improving discrimination beyond simple cosine similarity.  
  - **Dimensionality Reduction**: Apply techniques like Principal Component Analysis (PCA) to reduce the dimensionality of embeddings, potentially improving computational efficiency.  
   
- **User Feedback Mechanisms**:  
  - **Interactive Corrections**: Implement a system where users can correct misidentifications, allowing the system to adjust embeddings or thresholds accordingly.  
  - **Continuous Learning**: Enable the system to learn from new data over time, refining embeddings and improving accuracy with continued use.  
   
#### Summary  
   
As more speakers are added to the system:  
   
- **Variability Increases**: The diversity of voice characteristics grows, and the likelihood of speakers having similar vocal attributes rises.  
- **Similarity Scores Become Less Distinct**: Embeddings for different speakers may yield higher cosine similarity scores, making it challenging to differentiate them using a fixed threshold.  
- **Challenges in Accuracy and Performance**: Maintaining high identification accuracy becomes more difficult, and computational demands increase, potentially affecting real-time operation.  
- **Need for Advanced Techniques**: To ensure the system remains effective, especially when tracking many speakers, more sophisticated methods and optimizations are necessary beyond basic cosine similarity comparisons.  
   
By understanding these challenges and implementing appropriate mitigation strategies, it's possible to enhance the system's scalability and maintain accuracy even as the number of speakers increases.

---

   
## The `SpeakerIdentifier` Class  
   
The `SpeakerIdentifier` class encapsulates the functionality for enrollment, real-time audio processing, speaker identification, and transcription.  
   
Key methods include:  
   
- **`__init__`**: Initializes models, audio settings, and parameters.  
- **`record_enrollment`**: Records samples of the user's voice for enrollment.  
- **`extract_voiced_audio`**: Extracts voiced frames from audio data using VAD.  
- **`create_embedding`**: Creates voice embeddings from audio data.  
- **`vad_collector`**: Collects voiced frames from the audio queue.  
- **`process_segment`**: Processes a voiced segment for speaker identification and transcription.  
- **`compare_embeddings`**: Computes cosine similarity between two embeddings.  
- **`transcribe_audio`**: Transcribes audio data using Vosk.  
- **`audio_callback`**: Callback function for the audio stream.  
- **`start_listening`**: Starts the real-time audio stream and processing.  
   
---  
   
## Setting Up the Environment  
   
### Installing Dependencies  
   
Install the required Python packages using `pip`:  
   
```bash  
pip3 install numpy sounddevice webrtcvad vosk torch speechbrain  
```  
   
Ensure you have [PyTorch](https://pytorch.org/get-started/locally/) installed with CUDA support if you have a compatible GPU.  
   
### Downloading Models  
   
1. **Vosk Model**: Download the Vosk English model and place it in the `models` directory.  
  
   ```bash  
   mkdir models  
   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip  
   unzip vosk-model-small-en-us-0.15.zip -d models/  
   ```  
   
2. **SpeechBrain Model**: The script automatically downloads the SpeechBrain ECAPA-TDNN model and saves it to `pretrained_models/spkrec-ecapa-voxceleb`.  
   
---  
   
## Running the Script  
   
1. **Enrollment**: The script first prompts you to record enrollment samples.  
  
   ```python  
   speaker_id.record_enrollment(num_samples=3, duration=5)  
   ```  
  
   - **`num_samples`**: Number of samples to record.  
   - **`duration`**: Duration of each sample in seconds.  
   
2. **Start Listening**: Begins real-time processing.  
  
   ```python  
   speaker_id.start_listening()  
   ```  
  
   - **Interrupt**: Press `Ctrl+C` to stop the script.  
   
**Example Output**:  
   
```  
Recording enrollment sample 1/3 for 5 seconds...  
Recording enrollment sample 2/3 for 5 seconds...  
Recording enrollment sample 3/3 for 5 seconds...  
Averaged enrollment embedding created.  
Starting real-time processing. Press Ctrl+C to stop.  
[0:00:05 - 0:00:10] [Me] Hello, this is a test.  
[0:00:15 - 0:00:20] [Speaker 1] Hi there, how are you?  
[0:00:25 - 0:00:30] [Me] I'm fine, thank you!  
```  
   
---  
   
## Conclusion  
   
This script demonstrates how to combine powerful speech processing tools to perform real-time speaker identification and transcription. While it provides a solid foundation, there is room for enhancements:  
   
- **Noise Robustness**: Implement noise reduction techniques to handle noisy environments.  
- **Language Support**: Extend to support multiple languages by downloading appropriate models.  
- **GUI Integration**: Create a user-friendly interface for broader accessibility.  
   
---  
   
## References  
   
- [Vosk Speech Recognition Toolkit](https://alphacephei.com/vosk/)  
- [WebRTC Voice Activity Detection](https://github.com/wiseman/py-webrtcvad)  
- [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)  

---  
   
## Appendix: Fine-Tuning SpeechBrain for Improved Speaker Identification  
   
While the pre-trained SpeechBrain ECAPA-TDNN model offers robust performance for general speaker identification tasks, fine-tuning the model on your specific dataset can enhance accuracy and adaptability, particularly in specialized environments or with a limited set of speakers.  
   
### Why Fine-Tune SpeechBrain?  
   
- **Domain Adaptation**: Tailor the model to specific acoustic environments, such as office spaces, outdoor settings, or telecommunication channels.  
- **Speaker Specificity**: Improve discrimination between a known set of speakers, which is beneficial for meetings or collaborative settings.  
- **Enhanced Robustness**: Address challenges like background noise, varied microphone qualities, and speaker accents.  
   
### Steps for Fine-Tuning  
   
1. **Collect and Prepare Data**:  
   - Gather audio recordings of the target speakers in the relevant environment.  
   - Ensure recordings are of sufficient length (preferably several minutes per speaker).  
   - Label each audio file with the corresponding speaker identity.  
   
2. **Set Up the Environment**:  
   - Clone the SpeechBrain repository and navigate to the speaker recognition recipe:  
     ```bash  
     git clone https://github.com/speechbrain/speechbrain.git  
     cd speechbrain/recipes/VoxCeleb/SpeakerRec  
     ```  
   - Install any additional requirements specified in the repository.  
   
3. **Configure the Training Parameters**:  
   - Adjust the `hparams.yaml` file to point to your dataset and modify training parameters as needed.  
   - Ensure the `data_folder` parameter reflects the path to your prepared dataset.  
   
4. **Initiate Fine-Tuning**:  
   - Run the training script with your custom hyperparameters:  
     ```bash  
     python train_speaker_embeddings.py hparams/your_custom_hparams.yaml  
     ```  
   - Monitor training to prevent overfitting, possibly by using early stopping based on validation loss.  
   
5. **Replace the Pre-Trained Model in Your Script**:  
   - After fine-tuning, update the model loading in your script to use the fine-tuned model:  
     ```python  
     self.encoder = EncoderClassifier.from_hparams(  
         source="path/to/your/fine_tuned_model",  
         run_opts={"device": self.device},  
     )  
     ```  
   - Ensure the `source` parameter points to the directory containing your fine-tuned model.  
   
### Tips for Effective Fine-Tuning  
   
- **Data Augmentation**: Use techniques like noise addition, speed perturbation, and reverberation to augment your dataset, enhancing the model's robustness.  
- **Balanced Dataset**: Ensure that all target speakers are represented equally in the training data to prevent bias.  
- **Validation Set**: Keep a separate validation set to objectively measure performance improvements during fine-tuning.  
   
### Evaluating Performance  
   
- **Cosine Similarity Threshold**: After fine-tuning, you might need to adjust the `THRESHOLD` value in your script to align with the new model's characteristics.  
- **Testing**: Conduct tests with known speakers to assess identification accuracy and adjust accordingly.  
   
By fine-tuning SpeechBrain on your specific data, you can significantly improve the performance of your real-time speaker identification system, making it more accurate and reliable for your particular use case.  
   
---

# CODE

```
import os  
import sys  
import threading  
import numpy as np  
import sounddevice as sd  
from datetime import timedelta  
import webrtcvad  
import json  
from vosk import Model, KaldiRecognizer  
import queue  
import time  
import logging  
  
import torch  
from speechbrain.pretrained import EncoderClassifier  
  
# Set up logging  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
class SpeakerIdentifier:  
    """  
    A class to perform real-time speaker identification and transcription.  
    """  
  
    # Constants for easy tweaking  
    MODEL_NAME = "models/vosk-model-small-en-us-0.15"  # Vosk model directory  
    CHANNELS = 1                                      # Number of audio channels (mono)  
    FRAME_DURATION_MS = 30                            # Frame size in milliseconds  
    VAD_AGGRESSIVENESS = 2                            # VAD aggressiveness (0-3)  
    THRESHOLD = 0.6                                   # Similarity threshold for speaker identification  
    MIN_SEGMENT_DURATION = 1.0                        # Minimum duration of a segment in seconds  
  
    def __init__(self):  
        # Adjust sample rate based on the default input device  
        try:  
            default_input_device = sd.query_devices(kind='input')  
            default_sample_rate = int(default_input_device['default_samplerate'])  
            self.sample_rate = default_sample_rate if default_sample_rate else 16000  
        except Exception as e:  
            logger.error(f"Could not determine default sample rate: {e}")  
            self.sample_rate = 16000  
  
        if self.sample_rate != 16000:  
            logger.warning(f"Default sample rate is {self.sample_rate} Hz. Adjusting to match the microphone's capabilities.")  
  
        # Initialize Vosk model  
        if not os.path.exists(self.MODEL_NAME):  
            logger.error(f"Vosk model '{self.MODEL_NAME}' not found. Please download and place it in the script directory.")  
            sys.exit(1)  
        logger.info("Loading Vosk model...")  
        self.model = Model(self.MODEL_NAME)  
        logger.info("Vosk model loaded.")  
  
        # Initialize the voice encoder using SpeechBrain's ECAPA-TDNN model  
        logger.info("Initializing voice encoder...")  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.encoder = EncoderClassifier.from_hparams(  
            source="speechbrain/spkrec-ecapa-voxceleb",  
            savedir="models/spkrec-ecapa-voxceleb",  
            run_opts={"device": self.device},  
        )  
        logger.info("Voice encoder initialized.")  
  
        # Initialize WebRTC VAD  
        self.vad = webrtcvad.Vad(self.VAD_AGGRESSIVENESS)  
  
        # Buffer to hold audio frames  
        self.audio_queue = queue.Queue()  
        self.running = True  
  
        # Enrollment embedding  
        self.my_embedding = None  
  
        # For timestamp calculations  
        self.recording_start_time = None  
  
        # Dictionary to store other speakers' embeddings and assign IDs  
        self.other_speakers = {}  
        self.next_speaker_id = 1  
  
    def record_enrollment(self, num_samples=3, duration=5):  
        """  
        Records multiple audio samples for enrollment and creates an averaged voice embedding for the user.  
        """  
        embeddings = []  
        for i in range(num_samples):  
            logger.info(f"Recording enrollment sample {i + 1}/{num_samples} for {duration} seconds...")  
            try:  
                audio_data = sd.rec(int(duration * self.sample_rate),  
                                    samplerate=self.sample_rate,  
                                    channels=self.CHANNELS,  
                                    dtype='int16')  
                sd.wait()  
                audio_data = np.squeeze(audio_data)  
                # Extract voiced audio using VAD  
                voiced_audio = self.extract_voiced_audio(audio_data)  
                # Create embedding from voiced audio  
                embedding = self.create_embedding(voiced_audio)  
                embeddings.append(embedding)  
            except Exception as e:  
                logger.error(f"Error during enrollment recording: {e}")  
                sys.exit(1)  
        if embeddings:  
            # Average the embeddings  
            self.my_embedding = np.mean(embeddings, axis=0)  
            # Normalize the averaged embedding  
            self.my_embedding = self.my_embedding / np.linalg.norm(self.my_embedding)  
            logger.info("Averaged enrollment embedding created.")  
        else:  
            logger.error("No embeddings were created during enrollment.")  
            sys.exit(1)  
  
    def extract_voiced_audio(self, audio_data):  
        """  
        Extracts voiced frames from the audio data using VAD.  
        """  
        frame_size = int(self.FRAME_DURATION_MS / 1000 * self.sample_rate)  # Frame size in samples  
        frames = [audio_data[i:i + frame_size] for i in range(0, len(audio_data), frame_size)]  
        voiced_frames = []  
        for frame in frames:  
            if len(frame) < frame_size:  
                # Pad the frame if it's shorter than the expected frame size  
                frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')  
            frame_bytes = frame.tobytes()  
            is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)  
            if is_speech:  
                voiced_frames.extend(frame)  
        voiced_audio = np.array(voiced_frames, dtype=np.int16)  
        return voiced_audio  
  
    def create_embedding(self, audio_data):  
        """  
        Creates a voice embedding from audio data using SpeechBrain's encoder.  
        """  
        # Check if the audio data is long enough  
        if len(audio_data) < self.sample_rate * self.MIN_SEGMENT_DURATION:  
            raise ValueError("Audio segment is too short for embedding.")  
  
        # Convert audio data to float32 numpy array and normalize  
        audio_float32 = audio_data.astype(np.float32) / 32768.0  
  
        # Convert numpy array to PyTorch tensor and add batch dimension  
        audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(self.device)  
  
        # Generate the embedding  
        with torch.no_grad():  
            embeddings = self.encoder.encode_batch(audio_tensor)  
        embedding = embeddings.squeeze().cpu().numpy()  
  
        # Normalize the embedding  
        embedding = embedding / np.linalg.norm(embedding)  
        return embedding  
  
    def vad_collector(self):  
        """  
        Collects voiced frames from the audio queue using VAD.  
        """  
        frame_duration = self.FRAME_DURATION_MS / 1000.0  
        num_padding_frames = int(0.5 / frame_duration)  # 0.5 seconds of padding  
        ring_buffer = []  
        triggered = False  
  
        voiced_frames = []  
  
        while self.running:  
            try:  
                frame = self.audio_queue.get(timeout=1)  
            except queue.Empty:  
                continue  
  
            is_speech = self.vad.is_speech(frame, self.sample_rate)  
            current_time = time.time()  
  
            if not triggered:  
                ring_buffer.append((frame, is_speech, current_time))  
                num_voiced = len([f for f, speech, _ in ring_buffer if speech])  
                if num_voiced > 0.9 * len(ring_buffer):  
                    triggered = True  
                    voiced_frames.extend([f for f, _, _ in ring_buffer])  
                    start_time = ring_buffer[0][2]  
                    ring_buffer = []  
                elif len(ring_buffer) > num_padding_frames:  
                    ring_buffer.pop(0)  
            else:  
                voiced_frames.append(frame)  
                ring_buffer.append((frame, is_speech, current_time))  
                num_unvoiced = len([f for f, speech, _ in ring_buffer if not speech])  
                if num_unvoiced > 0.9 * len(ring_buffer):  
                    end_time = ring_buffer[-1][2]  
                    triggered = False  
                    self.process_segment(voiced_frames, start_time, end_time)  
                    ring_buffer = []  
                    voiced_frames = []  
                elif len(ring_buffer) > num_padding_frames:  
                    ring_buffer.pop(0)  
  
        # Process any remaining frames  
        if voiced_frames:  
            end_time = time.time()  
            self.process_segment(voiced_frames, start_time, end_time)  
  
    def process_segment(self, frames, start_time, end_time):  
        """  
        Processes a voiced segment: speaker identification and transcription.  
        """  
        # Initialize recording start time if not set  
        if self.recording_start_time is None:  
            self.recording_start_time = start_time  
  
        # Convert frames to numpy array  
        segment = b''.join(frames)  
        segment_audio = np.frombuffer(segment, dtype=np.int16)  
  
        # Ignore short segments  
        duration = end_time - start_time  
        if duration < self.MIN_SEGMENT_DURATION:  
            return  
  
        # Speaker identification  
        try:  
            # Create an embedding for the current segment  
            segment_embedding = self.create_embedding(segment_audio)  
  
            # Compare with the enrollment embedding  
            similarity_to_me = self.compare_embeddings(self.my_embedding, segment_embedding)  
  
            # Log the similarity score  
            logger.info(f"Similarity to 'Me': {similarity_to_me:.4f}")  
  
            if similarity_to_me >= self.THRESHOLD:  
                speaker_label = "Me"  
            else:  
                # Check against other speakers  
                found = False  
                for speaker_id, embedding in self.other_speakers.items():  
                    similarity = self.compare_embeddings(embedding, segment_embedding)  
                    # Log the similarity score  
                    logger.info(f"Similarity to Speaker {speaker_id}: {similarity:.4f}")  
                    if similarity >= self.THRESHOLD:  
                        speaker_label = f"Speaker {speaker_id}"  
                        found = True  
                        break  
                if not found:  
                    # Assign new speaker ID  
                    speaker_label = f"Speaker {self.next_speaker_id}"  
                    self.other_speakers[self.next_speaker_id] = segment_embedding  
                    logger.info(f"Assigned new speaker ID: {self.next_speaker_id}")  
                    self.next_speaker_id += 1  
        except ValueError as e:  
            logger.error(f"Error during speaker identification: {e}")  
            speaker_label = "Unknown"  
  
        # Transcription  
        transcript = self.transcribe_audio(segment_audio)  
  
        # Display the result  
        relative_start_time = start_time - self.recording_start_time  
        relative_end_time = end_time - self.recording_start_time  
        start_td = str(timedelta(seconds=int(relative_start_time)))  
        end_td = str(timedelta(seconds=int(relative_end_time)))  
        logger.info(f"[{start_td} - {end_td}] [{speaker_label}] {transcript}")  
  
    def compare_embeddings(self, embedding1, embedding2):  
        """  
        Computes the cosine similarity between two embeddings.  
        """  
        # Ensure embeddings are flattened to 1D arrays  
        embedding1 = embedding1.flatten()  
        embedding2 = embedding2.flatten()  
  
        # Since embeddings are normalized, the dot product is the cosine similarity  
        similarity = np.dot(embedding1, embedding2)  
        return similarity  
  
    def transcribe_audio(self, audio_data):  
        """  
        Transcribes audio data using Vosk.  
        """  
        rec = KaldiRecognizer(self.model, self.sample_rate)  
        rec.SetWords(True)  
  
        audio_bytes = audio_data.tobytes()  
        rec.AcceptWaveform(audio_bytes)  
        res = rec.Result()  
        res_json = json.loads(res)  
        transcript = res_json.get('text', '')  
        return transcript  
  
    def audio_callback(self, indata, frames, time_info, status):  
        """  
        Callback function for real-time audio processing.  
        """  
        if status:  
            logger.warning(f"Audio status: {status}")  
  
        # Convert audio input to bytes and put into the queue  
        self.audio_queue.put(indata.tobytes())  
  
    def start_listening(self):  
        """  
        Starts the real-time audio stream and voice activity detection.  
        """  
        self.running = True  
        logger.info("Starting real-time processing. Press Ctrl+C to stop.")  
  
        # Start VAD collector in a separate thread  
        vad_thread = threading.Thread(target=self.vad_collector)  
        vad_thread.daemon = True  
        vad_thread.start()  
  
        # Start audio stream  
        try:  
            with sd.InputStream(  
                samplerate=self.sample_rate,  
                channels=self.CHANNELS,  
                dtype='int16',  
                blocksize=int(self.sample_rate * (self.FRAME_DURATION_MS / 1000.0)),  
                callback=self.audio_callback  
            ):  
                try:  
                    while self.running:  
                        time.sleep(0.1)  
                except KeyboardInterrupt:  
                    logger.info("Stopping...")  
                    self.running = False  
        except Exception as e:  
            logger.error(f"Error during audio streaming: {e}")  
            self.running = False  
  
        # Wait for the audio queue to empty  
        while not self.audio_queue.empty():  
            time.sleep(0.1)  
        vad_thread.join()  
  
def main():  
    speaker_id = SpeakerIdentifier()  
    try:  
        speaker_id.record_enrollment(num_samples=3, duration=5)  
        speaker_id.start_listening()  
    except KeyboardInterrupt:  
        logger.info("Interrupted by user.")  
    except Exception as e:  
        logger.error(f"An error occurred: {e}")  
  
if __name__ == "__main__":  
    main()  

```
