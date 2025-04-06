variable "project_id" {
  description = "The Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "The region to deploy to"
  type        = string
  default     = "asia-northeast1"
}

variable "secrets" {
  description = "Map of environment variables to their corresponding secrets"
  type = map(object({
    secret_id = string
  }))
  default = {
    SLACK_BOT_TOKEN = {
      secret_id = "llm_slack_bot"
    }
    SLACK_APP_TOKEN = {
      secret_id = "llm_slack_app_token"
    }
    SLACK_SIGNING_SECRET = {
      secret_id = "llm_slack_bot_sign"
    }
    OPENAI_API_KEY = {
      secret_id = "openai"
    }
    ANTHROPIC_API_KEY = {
      secret_id = "anthropic_api_key"
    }
    GOOGLE_API_KEY = {
      secret_id = "google_api_key"
    }
  }
}
