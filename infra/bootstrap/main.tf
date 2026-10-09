# Step 2 bootstrap: API enablement and billing budget alerts for the
# remember-or-retrieve project. Nothing here costs money.
#
# Auth: export GOOGLE_OAUTH_ACCESS_TOKEN from the named gcloud configuration, so the
# default gcloud account and Application Default Credentials are never used:
#   $env:GOOGLE_OAUTH_ACCESS_TOKEN = gcloud auth print-access-token --configuration=remember-or-retrieve
# The billing account ID is supplied via TF_VAR_billing_account (never committed).

terraform {
  required_version = ">= 1.9"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

variable "project_id" {
  type    = string
  default = "remember-or-retrieve"
}

variable "billing_account" {
  type        = string
  description = "Billing account ID, e.g. via TF_VAR_billing_account. Not committed."
}

variable "currency_code" {
  type        = string
  default     = "CAD"
  description = "Must match the billing account's currency (budget creation fails otherwise)."
}

variable "usd_to_cad" {
  type        = number
  default     = 1.40
  description = "Assumed exchange rate used to express the USD budget in the account currency."
}

provider "google" {
  project               = var.project_id
  user_project_override = true
  billing_project       = var.project_id
}

locals {
  apis = [
    "compute.googleapis.com",
    "container.googleapis.com",
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "billingbudgets.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "serviceusage.googleapis.com",
    "cloudquotas.googleapis.com",
    "cloudbilling.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  service            = each.value
  disable_on_destroy = false
}

data "google_project" "this" {}

# The billing account is in CAD, so the USD targets are converted at var.usd_to_cad.
# Two budgets so each threshold is its own alert: the $20 target and the $50 hard cap.
# Alerts email the billing account admins. They warn; they do not stop spend.
resource "google_billing_budget" "target" {
  billing_account = var.billing_account
  display_name    = "remember-or-retrieve target ($20)"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }
  amount {
    specified_amount {
      currency_code = var.currency_code
      units         = tostring(ceil(20 * var.usd_to_cad))
    }
  }
  threshold_rules { threshold_percent = 0.5 }
  threshold_rules { threshold_percent = 1.0 }

  depends_on = [google_project_service.apis]
}

resource "google_billing_budget" "cap" {
  billing_account = var.billing_account
  display_name    = "remember-or-retrieve hard cap ($50)"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }
  amount {
    specified_amount {
      currency_code = var.currency_code
      units         = tostring(ceil(50 * var.usd_to_cad))
    }
  }
  threshold_rules { threshold_percent = 0.5 }
  threshold_rules { threshold_percent = 0.9 }
  threshold_rules { threshold_percent = 1.0 }

  depends_on = [google_project_service.apis]
}
