import ollama

class RobotBrain:
    def __init__(self, model_name="llama3.2:1b"):
        print("[System] Initializing Brain (Ollama)...")
        self.model_name = model_name
        self.system_prompt = """
        You are a helpful robotic assistant running locally on hardware.
        Answer in exactly 1 or 2 short sentences.
        Never use lists, bullet points, asterisks, or emojis. 
        Speak naturally like a human.
        """
        self.chat_history = [{'role': 'system', 'content': self.system_prompt}]

    def think(self, user_text):
        print("⏳ Thinking...")
        self.chat_history.append({'role': 'user', 'content': user_text})
        
        response = ollama.chat(
            model=self.model_name,
            messages=self.chat_history
        )
        
        ai_text = response['message']['content']
        self.chat_history.append({'role': 'assistant', 'content': ai_text})
        
        # Prune old history to save memory while keeping the system prompt
        if len(self.chat_history) > 11:
            self.chat_history.pop(1) 
            self.chat_history.pop(1)
            
        print(f"🤖 Robot: {ai_text}")
        return ai_text