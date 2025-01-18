from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from google.cloud import firestore
import os
from datetime import datetime
import openai
from anthropic import Anthropic
import google.generativeai as genai
from dotenv import load_dotenv
from abc import ABC, abstractmethod

load_dotenv()

# Initialize Slack app
app = App(token=os.environ["SLACK_BOT_TOKEN"])

# Initialize AI clients
openai.api_key = os.environ["OPENAI_API_KEY"]
anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize Firestore
db = firestore.Client()

def get_conversation_history(channel_id, thread_ts):
    """Get conversation history from Firestore"""
    try:
        doc_ref = db.collection('conversations').document(f"{channel_id}_{thread_ts}")
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            if data and 'messages' in data:
                return data['messages']
        return []
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        return []

def save_conversation(channel_id, thread_ts, messages):
    """Save conversation to Firestore"""
    doc_ref = db.collection('conversations').document(f"{channel_id}_{thread_ts}")
    doc_ref.set({
        'messages': messages,
        'updated_at': datetime.now()
    })

def update_message(client, channel_id, message_ts, text):
    """Update Slack message with accumulated response"""
    client.chat_update(
        channel=channel_id,
        ts=message_ts,
        text=text
    )

class AIHandler(ABC):
    def __init__(self, channel_id, message_ts, client):
        self.channel_id = channel_id
        self.message_ts = message_ts
        self.client = client
        self.buffer = ""
        self.accumulated_response = ""
        
    def handle_chunk(self, chunk_text):
        """Handle streaming chunk and update message"""
        self.buffer += chunk_text
        self.accumulated_response += chunk_text
        
        if len(self.buffer) >= 20:
            update_message(self.client, self.channel_id, self.message_ts, self.accumulated_response)
            self.buffer = ""
    
    def finalize_response(self):
        """Send final update if needed"""
        if self.accumulated_response:
            update_message(self.client, self.channel_id, self.message_ts, self.accumulated_response)
        return self.accumulated_response
    
    @abstractmethod
    def process_stream(self, text, history):
        """Process the stream for specific AI service"""
        pass

class OpenAIHandler(AIHandler):
    def process_stream(self, text, history):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": text})
        
        stream = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                self.handle_chunk(chunk.choices[0].delta.content)
        
        return self.finalize_response()

class ClaudeHandler(AIHandler):
    def process_stream(self, text, history):
        messages = []
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        messages.append({"role": "user", "content": text})
        
        with anthropic.messages.stream(
            model="claude-3-sonnet-20240229",
            messages=messages
        ) as stream:
            for chunk in stream:
                if chunk.type == "content_block_delta":
                    self.handle_chunk(chunk.delta.text)
        
        return self.finalize_response()

class GeminiHandler(AIHandler):
    def process_stream(self, text, history):
        model = genai.GenerativeModel('gemini-pro')
        chat = model.start_chat(history=[
            {"role": msg["role"], "parts": [msg["content"]]} for msg in history
        ])
        
        response = chat.send_message(text, stream=True)
        for chunk in response:
            if chunk.text:
                self.handle_chunk(chunk.text)
        
        return self.finalize_response()

def get_ai_handler(ai_type, channel_id, message_ts, client):
    """Factory function to get appropriate AI handler"""
    handlers = {
        "openai": OpenAIHandler,
        "claude": ClaudeHandler,
        "gemini": GeminiHandler
    }
    handler_class = handlers.get(ai_type, OpenAIHandler)
    return handler_class(channel_id, message_ts, client)

def process_message(event, say, is_mention=False):
    """Process message and route to appropriate AI service"""
    text = event.get("text", "")
    channel_id = event.get("channel")
    thread_ts = event.get("thread_ts", event.get("ts"))
    
    # Remove bot mention from text if it's a mention
    if is_mention:
        text = ' '.join(text.split()[1:])
    
    # Get conversation history
    history = get_conversation_history(channel_id, thread_ts)
    
    # Send initial message
    initial_response = say(text="思考中...", thread_ts=thread_ts)
    message_ts = initial_response['ts']
    
    # Determine which AI to use
    ai_type = "openai"  # default
    for ai in ["openai", "claude", "gemini"]:
        if ai in text.lower():
            ai_type = ai
            break
    
    # Get appropriate handler and process message
    handler = get_ai_handler(ai_type, channel_id, message_ts, app.client)
    final_response = handler.process_stream(text, history)
    
    # Save the updated conversation
    history.extend([
        {"role": "user", "content": text},
        {"role": "assistant", "content": final_response}
    ])
    save_conversation(channel_id, thread_ts, history)

def handle_mention(event, say):
    """Handle mentions and route to appropriate AI service"""
    process_message(event, say, is_mention=True)

def handle_message(event, say):
    """Handle messages in threads"""
    # Ignore messages from bots
    if event.get("bot_id"):
        return "OK"
        
    # Only process messages in threads
    if not event.get("thread_ts"):
        return "OK"
        
    # Process the message
    process_message(event, say)

def just_ack(ack):
    ack()
    
app.event("app_mention")(ack=just_ack, lazy=[handle_mention])
app.event("message")(ack=just_ack, lazy=[handle_message])

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
