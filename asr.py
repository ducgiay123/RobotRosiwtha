import sherpa_onnx
import sounddevice as sd
import time

class RobotEars:
    def __init__(self, model_dir):
        print("[System] Initializing Ears (ASR)...")
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{model_dir}/tokens.txt",
            encoder=f"{model_dir}/encoder-epoch-99-avg-1.int8.onnx",
            decoder=f"{model_dir}/decoder-epoch-99-avg-1.int8.onnx",
            joiner=f"{model_dir}/joiner-epoch-99-avg-1.int8.onnx",
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
        )
        self.sample_rate = 16000
        self.samples_per_read = int(0.1 * self.sample_rate)

    def listen_for_command(self):
        stream = self.recognizer.create_stream()
        last_result = ""
        last_update_time = time.time()
        
        print("\n🟢 Listening... (Speak now)")
        
        with sd.InputStream(channels=1, dtype="float32", samplerate=self.sample_rate) as s:
            while True:
                samples, _ = s.read(self.samples_per_read)
                stream.accept_waveform(self.sample_rate, samples.reshape(-1))
                
                while self.recognizer.is_ready(stream):
                    self.recognizer.decode_stream(stream)
                
                result = self.recognizer.get_result(stream)
                
                if last_result != result:
                    last_result = result
                    last_update_time = time.time()
                    print(f"\r🗣️ You: {result}", end="", flush=True)
                
                # Custom 1.5-second silence endpoint trigger
                if (time.time() - last_update_time) > 1.5 and result.strip():
                    print() # Move to new line
                    return result