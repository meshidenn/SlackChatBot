import pytest
from unittest.mock import Mock, patch
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
        print(f"Mock client collection call: {mock_db.collection.call_args}")
        print(f"Mock document call: {mock_collection.document.call_args}")
        print(f"Mock doc get call: {mock_document.get.call_args}")
        print(f"Mock to_dict return value: {mock_doc.to_dict.return_value}")
        print(f"Actual history: {history}")
        
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

def test_process_message(event_data, mock_say):
    with patch('main.handle_openai') as mock_openai, \
         patch('main.get_conversation_history') as mock_get_history, \
         patch('main.save_conversation') as mock_save:
        
        mock_get_history.return_value = []
        mock_openai.return_value = "test response"

        # Test normal message processing
        main.process_message(event_data, mock_say)

        mock_say.assert_called()
        mock_openai.assert_called_once()
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
