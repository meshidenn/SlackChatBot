terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Cloud Run service
resource "google_cloud_run_v2_service" "slack_bot" {
  name     = "slack-ai-bot"
  location = var.region

  template {
    containers {
      image = "gcr.io/${var.project_id}/slack-ai-bot:latest"
      
      ports {
        container_port = 8080
      }

      # Secret environment variables
      dynamic "env" {
        for_each = var.secrets
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value.secret_id
              version = "latest"
            }
          }
        }
      }
    }
  }
}

# IAM policy for Cloud Run service
resource "google_cloud_run_service_iam_member" "public_access" {
  location = google_cloud_run_v2_service.slack_bot.location
  service  = google_cloud_run_v2_service.slack_bot.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Note: The following secrets are assumed to already exist in Secret Manager
# If they don't exist, you'll need to create them separately or add their creation to this Terraform config
