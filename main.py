from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.models.blocks import ButtonElement, ActionsBlock
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

def save_conversation(channel_id, thread_ts, messages, model=None):
    """Save conversation to Firestore"""
    data = {
        'messages': messages,
        'updated_at': datetime.now()
    }
    if model:
        data['model'] = model
    
    doc_ref = db.collection('conversations').document(f"{channel_id}_{thread_ts}")
    doc_ref.set(data)

def get_thread_model(channel_id, thread_ts):
    """Get the model being used in the thread"""
    try:
        doc_ref = db.collection('conversations').document(f"{channel_id}_{thread_ts}")
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            if data and 'model' in data:
                return data['model']
    except Exception as e:
        print(f"Error getting thread model: {e}")
    return "openai"  # default to OpenAI

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
            max_tokens=1024,
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

def create_model_selection_blocks():
    """Create blocks with model selection buttons"""
    return [
        ActionsBlock(
            elements=[
                ButtonElement(
                    text="OpenAI GPT-4",
                    action_id="select_model_openai",
                    value="openai"
                ),
                ButtonElement(
                    text="Claude 3",
                    action_id="select_model_claude",
                    value="claude"
                ),
                ButtonElement(
                    text="Gemini Pro",
                    action_id="select_model_gemini",
                    value="gemini"
                )
            ]
        )
    ]

def get_model_name(ai_type):
    """Get human-readable model name"""
    model_names = {
        "openai": "OpenAI GPT-4",
        "claude": "Claude 3",
        "gemini": "Gemini Pro"
    }
    return model_names.get(ai_type, "Unknown Model")

def process_with_model(channel_id, thread_ts, text, history, message_ts, client, ai_type, save_model=False):
    """Process message with specified model"""
    model_name = get_model_name(ai_type)
    
    # Update message to show thinking state
    client.chat_update(
        channel=channel_id,
        ts=message_ts,
        text=f"思考中... (Using {model_name})",
        blocks=[]
    )
    
    # Process with selected model
    handler = get_ai_handler(ai_type, channel_id, message_ts, client)
    final_response = handler.process_stream(text, history)
    
    # Add model info to the response
    final_response_with_model = f"[{model_name}]\n{final_response}"
    
    # Update the message with the final response including model info
    client.chat_update(
        channel=channel_id,
        ts=message_ts,
        text=final_response_with_model
    )
    
    # Save the updated conversation
    history.extend([
        {"role": "user", "content": text},
        {"role": "assistant", "content": final_response}
    ])
    save_conversation(channel_id, thread_ts, history, ai_type if save_model else None)

def process_mention(event, say):
    """Process mention and show model selection"""
    text = ' '.join(event.get("text", "").split()[1:])  # Remove bot mention
    channel_id = event.get("channel")
    thread_ts = event.get("thread_ts", event.get("ts"))
    
    # Get conversation history
    history = get_conversation_history(channel_id, thread_ts)
    
    # Send initial message with model selection buttons
    initial_response = say(
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "使用するモデルを選択してください"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "OpenAI GPT-4"
                        },
                        "action_id": "select_model_openai",
                        "value": "openai"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Claude 3"
                        },
                        "action_id": "select_model_claude",
                        "value": "claude"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Gemini Pro"
                        },
                        "action_id": "select_model_gemini",
                        "value": "gemini"
                    }
                ]
            }
        ],
        thread_ts=thread_ts
    )
    message_ts = initial_response['ts']
    
    # Store the message info for later processing
    doc_ref = db.collection('pending_messages').document(message_ts)
    doc_ref.set({
        'channel_id': channel_id,
        'thread_ts': thread_ts,
        'text': text,
        'history': history,
        'created_at': datetime.now()
    })

def process_message(event, say):
    """Process thread message with thread's model"""
    # Ignore messages from bots
    if event.get("bot_id"):
        return "OK"
        
    # Only process messages in threads
    if not event.get("thread_ts"):
        return "OK"
    
    text = event.get("text", "")
    channel_id = event.get("channel")
    thread_ts = event.get("thread_ts")
    
    # Get conversation history and thread's model
    history = get_conversation_history(channel_id, thread_ts)
    model = get_thread_model(channel_id, thread_ts)
    
    # Send initial message
    initial_response = say(text="思考中...", thread_ts=thread_ts)
    message_ts = initial_response['ts']
    
    # Process with thread's model
    process_with_model(channel_id, thread_ts, text, history, message_ts, app.client, model)

def handle_mention(event, say):
    """Handle mentions and show model selection"""
    process_mention(event, say)

def handle_message(event, say):
    """Handle thread messages with default model"""
    process_message(event, say)

def handle_model_selection(ack, body, client):
    """Handle model selection button click"""
    ack()
    
    # Get message info
    message_ts = body["message"]["ts"]
    doc_ref = db.collection('pending_messages').document(message_ts)
    doc = doc_ref.get()
    
    if not doc.exists:
        return
    
    data = doc.to_dict()
    channel_id = data['channel_id']
    thread_ts = data['thread_ts']
    text = data['text']
    history = data['history']
    
    # Get selected model and process
    ai_type = body["actions"][0]["value"]
    process_with_model(channel_id, thread_ts, text, history, message_ts, client, ai_type, save_model=True)
    
    # Clean up pending message
    doc_ref.delete()

def just_ack(ack):
    ack()

# Register handlers
app.event("app_mention")(ack=just_ack, lazy=[handle_mention])
app.event("message")(ack=just_ack, lazy=[handle_message])
app.action("select_model_openai")(handle_model_selection)
app.action("select_model_claude")(handle_model_selection)
app.action("select_model_gemini")(handle_model_selection)

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
