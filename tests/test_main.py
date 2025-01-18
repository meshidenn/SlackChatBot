import pytest
from unittest.mock import Mock, patch, MagicMock
import main

@pytest.fixture
def mock_say():
    return Mock(return_value={"ts": "test_message_ts"})

@pytest.fixture
def event_data():
    return {
        "text": "Hello",
        "channel": "test_channel",
        "ts": "test_ts",
        "thread_ts": "test_thread_ts"
    }

@pytest.fixture
def mock_client():
    return Mock()

def test_get_conversation_history():
    with patch('main.db') as mock_db:
        # Setup mock chain
        mock_collection = Mock()
        mock_document = Mock()
        mock_doc = Mock()
        
        # Setup mock returns
        mock_db.collection = Mock(return_value=mock_collection)
        mock_collection.document = Mock(return_value=mock_document)
        mock_document.get = Mock(return_value=mock_doc)
        mock_doc.exists = True
        mock_doc.to_dict = Mock(return_value={
            "messages": [{
                "role": "user",
                "content": "test"
            }]
        })

        # Test existing conversation
        history = main.get_conversation_history("test_channel", "test_thread")
        
        # Verify the mock chain was called correctly
        mock_db.collection.assert_called_once_with('conversations')
        mock_collection.document.assert_called_once_with('test_channel_test_thread')
        mock_document.get.assert_called_once()
        mock_doc.to_dict.assert_called_once()
        
        # Verify the returned history
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "test"

        # Test non-existing conversation
        mock_doc.exists = False
        history = main.get_conversation_history("test_channel", "test_thread")
        assert len(history) == 0

def test_update_message():
    mock_client = Mock()
    main.update_message(mock_client, "test_channel", "test_ts", "test_text")
    mock_client.chat_update.assert_called_once_with(
        channel="test_channel",
        ts="test_ts",
        text="test_text"
    )

def test_openai_handler():
    mock_client = Mock()
    handler = main.OpenAIHandler("test_channel", "test_ts", mock_client)
    
    with patch('openai.chat.completions.create') as mock_create:
        # Mock streaming response
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "test response"
        mock_create.return_value = [mock_chunk]
        
        response = handler.process_stream("test input", [])
        
        assert response == "test response"
        mock_client.chat_update.assert_called()

def test_claude_handler():
    mock_client = Mock()
    handler = main.ClaudeHandler("test_channel", "test_ts", mock_client)
    
    with patch('main.anthropic.messages.stream') as mock_stream:
        # Mock streaming response
        mock_chunk = MagicMock()
        mock_chunk.type = "content_block_delta"
        mock_chunk.delta.text = "test response"
        mock_stream.return_value.__enter__.return_value = [mock_chunk]
        
        response = handler.process_stream("test input", [])
        
        assert response == "test response"
        mock_client.chat_update.assert_called()

def test_gemini_handler():
    mock_client = Mock()
    handler = main.GeminiHandler("test_channel", "test_ts", mock_client)
    
    with patch('google.generativeai.GenerativeModel') as MockModel:
        # Mock chat and response
        mock_chat = Mock()
        mock_chunk = Mock()
        mock_chunk.text = "test response"
        mock_chat.send_message.return_value = [mock_chunk]
        MockModel.return_value.start_chat.return_value = mock_chat
        
        response = handler.process_stream("test input", [])
        
        assert response == "test response"
        mock_client.chat_update.assert_called()

def test_process_message(event_data, mock_say):
    with patch('main.get_conversation_history') as mock_get_history, \
         patch('main.save_conversation') as mock_save, \
         patch('main.get_ai_handler') as mock_get_handler:
        
        mock_get_history.return_value = []
        mock_handler = Mock()
        mock_handler.process_stream.return_value = "test response"
        mock_get_handler.return_value = mock_handler

        # Test normal message processing
        main.process_message(event_data, mock_say)

        mock_say.assert_called()
        mock_handler.process_stream.assert_called_once()
        mock_save.assert_called_once()

def test_handle_message(event_data, mock_say):
    with patch('main.process_message') as mock_process:
        # Test thread message handling
        main.handle_message(event_data, mock_say)
        mock_process.assert_called_once()

        # Test bot message handling
        event_data["bot_id"] = "test_bot"
        result = main.handle_message(event_data, mock_say)
        assert result == "OK"

        # Test non-thread message handling
        event_data.pop("bot_id")
        event_data.pop("thread_ts")
        result = main.handle_message(event_data, mock_say)
        assert result == "OK"

def test_get_ai_handler():
    mock_client = Mock()
    
    # Test OpenAI handler
    handler = main.get_ai_handler("openai", "test_channel", "test_ts", mock_client)
    assert isinstance(handler, main.OpenAIHandler)
    
    # Test Claude handler
    handler = main.get_ai_handler("claude", "test_channel", "test_ts", mock_client)
    assert isinstance(handler, main.ClaudeHandler)
    
    # Test Gemini handler
    handler = main.get_ai_handler("gemini", "test_channel", "test_ts", mock_client)
    assert isinstance(handler, main.GeminiHandler)
    
    # Test default handler
    handler = main.get_ai_handler("unknown", "test_channel", "test_ts", mock_client)
    assert isinstance(handler, main.OpenAIHandler)
