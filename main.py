import json
import logging
import os
import re
from typing import Union
import hashlib
import hmac
import time

import functions_framework
import google.cloud.logging
import openai
from box import Box
from flask import Request, jsonify
from slack.signature import SignatureVerifier
from slack_bolt import App, context
from slack_bolt.adapter.google_cloud_functions import SlackRequestHandler

# Google Cloud Logging クライアント ライブラリを設定
logging_client = google.cloud.logging.Client()
logging_client.setup_logging(log_level=logging.DEBUG)

# 環境変数からシークレットを取得
slack_token = os.environ.get("SLACK_BOT_TOKEN")
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = openai_api_key

# FaaS で実行する場合、応答速度が遅いため process_before_response は True でなければならない
app = App(signing_secret=os.environ["SLACK_SECRET"], token=slack_token, process_before_response=True)
handler = SlackRequestHandler(app)

# Bot アプリにメンションしたイベントに対する応答
@app.event("app_mention")
def handle_app_mention_events(body: dict, say: context.say.say.Say):
    """アプリへのメンションに対する応答を生成する関数

    Args:
        body: HTTP リクエストのボディ
        say: 返信内容を Slack に送信
    """
    logging.debug(type(body))
    logging.debug(body)
    try:
        box = Box(body)
        user = box.event.user
        text = box.event.text
        only_text = re.sub("<@[a-zA-Z0-9]{11}>", "", text)
        # TODO: 単発の質問に返信するのみで、会話の履歴を参照する機能は未実装
        message = [{"role": "user", "content": only_text}]
        logging.debug(only_text)

        # OpenAI から AIモデルの回答を生成する
        (openai_response, total_tokens) = create_chat_completion(message)
        logging.debug(openai_response)
        logging.debug(f"total_tokens: {total_tokens}")

        # 課金額がわかりやすいよう消費されたトークンを返信に加える
        say(f"<@{user}> {openai_response}\n消費されたトークン:{total_tokens}")
    except Exception as e:
        logger.exception(f"Error handling app_mention event: {e}")
        say(f"エラーが発生しました。: {e}")


def create_chat_completion(messages: list) -> tuple[str, int]:
    """OpenAI API を呼び出して、質問に対する回答を生成する関数

    Args:
        messages: チャット内容のリスト

    Returns:
        GPT-3.5 の生成した回答内容
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        openai_response = response["choices"][0]["message"]["content"]
        total_tokens = response["usage"]["total_tokens"]
        return (openai_response, total_tokens)
    except Exception as e:
        logger.exception(f"Error calling OpenAI API: {e}")
        return ("OpenAI APIの呼び出しでエラーが発生しました。", 0)

# [START functions_verify_webhook]
def verify_slack_signature(request, signing_secret):
    request_body = request.get_data()  # Decodes received requests into request.data

    verifier = SignatureVerifier(signing_secret)

    if not verifier.is_valid_request(request.data, request.headers):
        raise ValueError("Invalid request/credentials.")
    
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    signature = request.headers.get("X-Slack-Signature")

    if not timestamp or not signature:
        return False

    # 5分以上前のリクエストは拒否
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False

    sig_basestring = f"v0:{timestamp}:{request_body.decode('utf-8')}"
    my_signature = hmac.new(
        signing_secret.encode("utf-8"),
        sig_basestring.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    my_signature = f"v0={my_signature}"

    return hmac.compare_digest(my_signature, signature)

# Cloud Functions で呼び出されるエントリポイント
@functions_framework.http
def slack_bot(request: Request):
    """slack のイベントリクエストを受信して各処理を実行する関数

    Args:
        request: Slack のイベントリクエスト

    Returns:
        SlackRequestHandler への接続
    """
    signing_secret = os.environ["SLACK_SECRET"]
    if not signing_secret:
        logging.error("SLACK_SIGNING_SECRET is not set in environment variables.")
        return "Internal Server Error", 500

    if not verify_slack_signature(request, signing_secret):
        logging.error("Invalid Slack signature")
        return "Invalid signature", 400
    header = request.headers
    logging.debug(f"header: {header}")
    body = request.get_json()
    logging.debug(f"body: {body}")

    # URL確認を通すとき
    if body.get("type") == "url_verification" or "challenge" in body:
        logging.info("url verification started")
        headers = {"Content-Type": "application/json"}
        res = json.dumps({"challenge": body["challenge"]})
        logging.debug(f"res: {res}")
        return (res, 200, headers)
    # 応答が遅いと Slack からリトライを何度も受信してしまうため、リトライ時は処理しない
    elif header.get("x-slack-retry-num"):
        logging.info("slack retry received")
        return {"statusCode": 200, "body": json.dumps({"message": "No need to resend"})}
    
    # handler への接続 class: flask.wrappers.Response
    try:
        logging.debug("Before handler.handle(request)")  # 追加
        response = handler.handle(request)
        logging.debug(f"After handler.handle(request): {response}")  # 追加
        return response
    except Exception as e:
        logging.exception(f"Exception in handler.handle: {e}") # 例外の詳細をログ出力
        return "Internal Server Error", 500 # 明示的に500エラーを返す
    # return handler.handle(request)