from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from google.cloud import firestore
import os
from datetime import datetime
import openai
from anthropic import Anthropic
import google.generativeai as genai
from dotenv import load_dotenv

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
    doc_ref = db.collection('conversations').document(f"{channel_id}_{thread_ts}")
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get('messages', [])
    return []

def save_conversation(channel_id, thread_ts, messages):
    """Save conversation to Firestore"""
    doc_ref = db.collection('conversations').document(f"{channel_id}_{thread_ts}")
    doc_ref.set({
        'messages': messages,
        'updated_at': datetime.now()
    })

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
    
    # Determine which AI to use based on text
    if "openai" in text.lower():
        response = handle_openai(text, history)
    elif "claude" in text.lower():
        response = handle_claude(text, history)
    elif "gemini" in text.lower():
        response = handle_gemini(text, history)
    else:
        # Default to OpenAI
        response = handle_openai(text, history)
    
    # Save the updated conversation
    history.extend([
        {"role": "user", "content": text},
        {"role": "assistant", "content": response}
    ])
    save_conversation(channel_id, thread_ts, history)
    
    # Send response in thread
    say(text=response, thread_ts=thread_ts)

# @app.event("app_mention")
def handle_mention(event, say):
    """Handle mentions and route to appropriate AI service"""
    process_message(event, say, is_mention=True)

# @app.event("message")
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

def handle_openai(text, history):
    """Handle OpenAI chat completion"""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": text})
    
    response = openai.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages
    )
    return response.choices[0].message.content

def handle_claude(text, history):
    """Handle Claude chat completion"""
    messages = []
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    messages.append({"role": "user", "content": text})
    
    response = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        messages=messages
    )
    return response.content[0].text

def handle_gemini(text, history):
    """Handle Gemini chat completion"""
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[
        {"role": msg["role"], "parts": [msg["content"]]} for msg in history
    ])
    response = chat.send_message(text)
    return response.text

def just_ack(ack):
    ack()
    
app.event("app_mention")(ack=just_ack, lazy=[handle_mention])
app.event("message")(ack=just_ack, lazy=[handle_message])

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
