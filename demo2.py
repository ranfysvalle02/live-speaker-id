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
