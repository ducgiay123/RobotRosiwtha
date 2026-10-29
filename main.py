import sys
from asr import RobotEars
from llm import RobotBrain
from tts import RobotMouth

def main():
    print("="*50)
    print("🚀 BOOTING MODULAR ROBOT PIPELINE")
    print("="*50)
    
    # 1. Instantiate the modules
    ears = RobotEars(model_dir="./models/sherpa-onnx-streaming-zipformer-en-2023-06-21")
    brain = RobotBrain(model_name="llama3.1:8b")
    mouth = RobotMouth(model_dir="./models/vits-piper-en_US-ryan-medium")
    
    print("\n✅ Boot sequence complete. Press Ctrl+C to shutdown.")
    
    try:
        while True:
            # Execution Pipeline
            user_command = ears.listen_for_command()
            ai_response = brain.think(user_command)
            mouth.speak(ai_response)
            
    except KeyboardInterrupt:
        print("\n\n[System] Shutting down robotic pipeline. Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    main()