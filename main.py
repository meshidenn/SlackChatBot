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
    
    # Send initial message
    initial_response = say(text="思考中...", thread_ts=thread_ts)
    message_ts = initial_response['ts']
    
    # Determine which AI to use based on text and get streaming response
    if "openai" in text.lower():
        final_response = handle_openai(text, history, channel_id, thread_ts, message_ts, app.client)
    elif "claude" in text.lower():
        final_response = handle_claude(text, history, channel_id, thread_ts, message_ts, app.client)
    elif "gemini" in text.lower():
        final_response = handle_gemini(text, history, channel_id, thread_ts, message_ts, app.client)
    else:
        # Default to OpenAI
        final_response = handle_openai(text, history, channel_id, thread_ts, message_ts, app.client)
    
    # Save the updated conversation
    history.extend([
        {"role": "user", "content": text},
        {"role": "assistant", "content": final_response}
    ])
    save_conversation(channel_id, thread_ts, history)

def handle_openai(text, history, channel_id, thread_ts, message_ts, client):
    """Handle OpenAI chat completion with streaming"""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": text})
    
    accumulated_response = ""
    buffer = ""
    
    stream = openai.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            buffer += chunk.choices[0].delta.content
            accumulated_response += chunk.choices[0].delta.content
            
            # バッファが一定の長さに達したら更新
            if len(buffer) >= 20:
                client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=accumulated_response
                )
                buffer = ""
    
    # 最終更新
    if accumulated_response:
        client.chat_update(
            channel=channel_id,
            ts=message_ts,
            text=accumulated_response
        )
    
    return accumulated_response

def handle_claude(text, history, channel_id, thread_ts, message_ts, client):
    """Handle Claude chat completion with streaming"""
    messages = []
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    messages.append({"role": "user", "content": text})
    
    accumulated_response = ""
    buffer = ""
    
    with anthropic.messages.stream(
        model="claude-3-sonnet-20240229",
        messages=messages
    ) as stream:
        for chunk in stream:
            if chunk.type == "content_block_delta":
                buffer += chunk.delta.text
                accumulated_response += chunk.delta.text
                
                # バッファが一定の長さに達したら更新
                if len(buffer) >= 20:
                    client.chat_update(
                        channel=channel_id,
                        ts=message_ts,
                        text=accumulated_response
                    )
                    buffer = ""
    
    # 最終更新
    if accumulated_response:
        client.chat_update(
            channel=channel_id,
            ts=message_ts,
            text=accumulated_response
        )
    
    return accumulated_response

def handle_gemini(text, history, channel_id, thread_ts, message_ts, client):
    """Handle Gemini chat completion with streaming"""
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[
        {"role": msg["role"], "parts": [msg["content"]]} for msg in history
    ])
    
    accumulated_response = ""
    buffer = ""
    
    response = chat.send_message(text, stream=True)
    for chunk in response:
        if chunk.text:
            buffer += chunk.text
            accumulated_response += chunk.text
            
            # バッファが一定の長さに達したら更新
            if len(buffer) >= 20:
                client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=accumulated_response
                )
                buffer = ""
    
    # 最終更新
    if accumulated_response:
        client.chat_update(
            channel=channel_id,
            ts=message_ts,
            text=accumulated_response
        )
    
    return accumulated_response

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
