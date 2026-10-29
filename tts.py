import sherpa_onnx
import sounddevice as sd
import numpy as np  # 1. Import numpy for array manipulation

class RobotMouth:
    def __init__(self, model_dir):
        print("[System] Initializing Mouth (TTS)...")
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=f"{model_dir}/en_US-ryan-medium.onnx", 
                    tokens=f"{model_dir}/tokens.txt",
                    data_dir=f"{model_dir}/espeak-ng-data",
                ),
                num_threads=1,
                debug=0,
                provider="cpu",
            )
        )
        self.tts = sherpa_onnx.OfflineTts(tts_config)

    def speak(self, text):
        audio = self.tts.generate(text)
        
        # 2. Create 0.5 seconds of blank audio (silence)
        silence_pad = np.zeros(int(0.5 * audio.sample_rate), dtype=np.float32)
        
        # 3. Attach the silence to the end of the speech
        final_audio = np.concatenate((audio.samples, silence_pad))
        
        # 4. Play the newly padded audio
        sd.play(final_audio, samplerate=audio.sample_rate)
        sd.wait()