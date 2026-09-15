terraform {
  required_version = ">= 1.6.0"
}

variable "environment" {
  type    = string
  default = "dev"
}

# Provider-specific resources are intentionally kept behind modules in a real deployment.
# This root module documents the environment boundary without hardcoding cloud credentials.

output "environment" {
  value = var.environment
}
