output "service_url" {
  value       = google_cloud_run_v2_service.slack_bot.uri
  description = "The URL of the deployed Cloud Run service"
}

output "service_name" {
  value       = google_cloud_run_v2_service.slack_bot.name
  description = "The name of the deployed Cloud Run service"
}

output "service_region" {
  value       = google_cloud_run_v2_service.slack_bot.location
  description = "The region where the service is deployed"
}
