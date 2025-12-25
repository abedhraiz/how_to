# Terraform Guide

## What is Terraform?

Terraform is an Infrastructure as Code (IaC) tool that allows you to define and provision infrastructure using declarative configuration files. It supports multiple cloud providers (AWS, Azure, GCP) and on-premises infrastructure.

## Prerequisites

- Terraform CLI installed
- Cloud provider account (AWS, Azure, GCP, etc.)
- Basic understanding of infrastructure concepts
- Text editor or IDE

## Installation

### Linux
```bash
# Download Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip

# Unzip
unzip terraform_1.6.0_linux_amd64.zip

# Move to PATH
sudo mv terraform /usr/local/bin/

# Verify installation
terraform version
```

### macOS
```bash
# Using Homebrew
brew tap hashicorp/tap
brew install hashicorp/tap/terraform

# Verify
terraform version
```

### Windows
```bash
# Using Chocolatey
choco install terraform

# Or download from: https://www.terraform.io/downloads
```

## Core Concepts

### 1. **Providers**
Plugins that interact with cloud providers or services (AWS, Azure, GCP, etc.).

### 2. **Resources**
Infrastructure components you want to create (EC2 instances, S3 buckets, etc.).

### 3. **State**
Terraform tracks the current state of your infrastructure in a state file.

### 4. **Modules**
Reusable Terraform configurations.

### 5. **Variables**
Input parameters for your configurations.

### 6. **Outputs**
Values returned after applying configuration.

### 7. **Data Sources**
Query existing infrastructure or external data.

## Basic Terraform Workflow

```bash
# Initialize working directory
terraform init

# Format configuration files
terraform fmt

# Validate configuration
terraform validate

# Preview changes
terraform plan

# Apply changes
terraform apply

# Destroy infrastructure
terraform destroy
```

## Basic Configuration

### Simple AWS EC2 Example

```hcl
# main.tf

# Configure the AWS Provider
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.0"
}

provider "aws" {
  region = "us-east-1"
}

# Create EC2 instance
resource "aws_instance" "web_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name        = "WebServer"
    Environment = "Development"
  }
}

# Output the public IP
output "instance_public_ip" {
  description = "Public IP of EC2 instance"
  value       = aws_instance.web_server.public_ip
}
```

## Terraform Configuration Syntax

### Providers

```hcl
# AWS Provider
provider "aws" {
  region     = "us-east-1"
  access_key = var.aws_access_key
  secret_key = var.aws_secret_key
}

# Azure Provider
provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

# Google Cloud Provider
provider "google" {
  project = "my-project-id"
  region  = "us-central1"
}

# Multiple Provider Instances
provider "aws" {
  alias  = "west"
  region = "us-west-2"
}

provider "aws" {
  alias  = "east"
  region = "us-east-1"
}
```

### Resources

```hcl
# Basic resource syntax
resource "resource_type" "resource_name" {
  argument1 = "value1"
  argument2 = "value2"
}

# AWS S3 Bucket
resource "aws_s3_bucket" "example" {
  bucket = "my-unique-bucket-name"

  tags = {
    Name        = "My bucket"
    Environment = "Dev"
  }
}

# AWS VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "main-vpc"
  }
}

# Resource with dependencies
resource "aws_subnet" "public" {
  vpc_id            = aws_vpc.main.id  # Reference another resource
  cidr_block        = "10.0.1.0/24"
  availability_zone = "us-east-1a"
}
```

### Variables

```hcl
# variables.tf

# String variable
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

# Number variable
variable "instance_count" {
  description = "Number of instances"
  type        = number
  default     = 1
}

# Boolean variable
variable "enable_monitoring" {
  description = "Enable detailed monitoring"
  type        = bool
  default     = false
}

# List variable
variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

# Map variable
variable "instance_tags" {
  description = "Tags for instances"
  type        = map(string)
  default = {
    Environment = "dev"
    Project     = "example"
  }
}

# Object variable
variable "instance_config" {
  description = "Instance configuration"
  type = object({
    instance_type = string
    ami           = string
    disk_size     = number
  })
}

# Using variables
resource "aws_instance" "example" {
  ami           = var.instance_config.ami
  instance_type = var.instance_config.instance_type
  tags          = var.instance_tags
}
```

### Variable Files

```hcl
# terraform.tfvars
region         = "us-west-2"
instance_count = 3
enable_monitoring = true

# dev.tfvars
environment = "development"
instance_type = "t2.micro"

# prod.tfvars
environment = "production"
instance_type = "t2.large"
```

```bash
# Use specific variable file
terraform apply -var-file="prod.tfvars"
```

### Outputs

```hcl
# outputs.tf

output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.web_server.id
}

output "instance_public_ip" {
  description = "Public IP address"
  value       = aws_instance.web_server.public_ip
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
  sensitive   = false
}

# Output with list
output "subnet_ids" {
  description = "IDs of subnets"
  value       = aws_subnet.public[*].id
}

# Output with map
output "instance_info" {
  description = "Instance information"
  value = {
    id        = aws_instance.web_server.id
    public_ip = aws_instance.web_server.public_ip
    type      = aws_instance.web_server.instance_type
  }
}
```

### Data Sources

```hcl
# Query existing AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
}

# Query existing VPC
data "aws_vpc" "existing" {
  id = "vpc-12345678"
}

# Query availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# Use data source in resource
resource "aws_instance" "example" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t2.micro"
}
```

## Terraform Commands

### Initialization
```bash
# Initialize directory
terraform init

# Upgrade providers
terraform init -upgrade

# Reconfigure backend
terraform init -reconfigure

# Get modules without backend initialization
terraform init -backend=false
```

### Planning and Applying
```bash
# Create execution plan
terraform plan

# Save plan to file
terraform plan -out=tfplan

# Plan with specific variable file
terraform plan -var-file="prod.tfvars"

# Apply changes
terraform apply

# Apply without confirmation
terraform apply -auto-approve

# Apply saved plan
terraform apply tfplan

# Apply with specific target
terraform apply -target=aws_instance.web_server
```

### Destroying
```bash
# Destroy all resources
terraform destroy

# Destroy without confirmation
terraform destroy -auto-approve

# Destroy specific resource
terraform destroy -target=aws_instance.web_server
```

### State Management
```bash
# List resources in state
terraform state list

# Show resource in state
terraform state show aws_instance.web_server

# Remove resource from state
terraform state rm aws_instance.web_server

# Move resource in state
terraform state mv aws_instance.old aws_instance.new

# Pull remote state
terraform state pull

# Push local state to remote
terraform state push

# Replace provider
terraform state replace-provider hashicorp/aws registry.terraform.io/hashicorp/aws
```

### Workspace Management
```bash
# List workspaces
terraform workspace list

# Create new workspace
terraform workspace new dev

# Switch workspace
terraform workspace select prod

# Delete workspace
terraform workspace delete dev

# Show current workspace
terraform workspace show
```

### Other Commands
```bash
# Format code
terraform fmt

# Recursively format
terraform fmt -recursive

# Validate configuration
terraform validate

# Show outputs
terraform output

# Show specific output
terraform output instance_id

# Refresh state
terraform refresh

# Import existing resource
terraform import aws_instance.example i-1234567890abcdef0

# Show current state
terraform show

# Create dependency graph
terraform graph | dot -Tsvg > graph.svg

# Console for testing expressions
terraform console
```

## Complete AWS Example

### Directory Structure
```
project/
├── main.tf
├── variables.tf
├── outputs.tf
├── terraform.tfvars
└── modules/
    └── vpc/
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```

### main.tf
```hcl
terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "my-terraform-state-bucket"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.project_name}-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

# Public Subnet
resource "aws_subnet" "public" {
  count             = length(var.public_subnet_cidrs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.public_subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]

  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public-subnet-${count.index + 1}"
  }
}

# Route Table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

# Route Table Association
resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Security Group
resource "aws_security_group" "web" {
  name_prefix = "${var.project_name}-web-sg"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_allowed_ips
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-web-sg"
  }
}

# EC2 Instance
resource "aws_instance" "web" {
  count         = var.instance_count
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  subnet_id     = aws_subnet.public[count.index % length(aws_subnet.public)].id

  vpc_security_group_ids = [aws_security_group.web.id]

  user_data = <<-EOF
              #!/bin/bash
              apt-get update
              apt-get install -y nginx
              systemctl start nginx
              systemctl enable nginx
              EOF

  tags = {
    Name        = "${var.project_name}-web-${count.index + 1}"
    Environment = var.environment
  }
}

# Data source for Ubuntu AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
}

# Data source for availability zones
data "aws_availability_zones" "available" {
  state = "available"
}
```

### variables.tf
```hcl
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "myapp"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t2.micro"
}

variable "instance_count" {
  description = "Number of EC2 instances"
  type        = number
  default     = 2
}

variable "ssh_allowed_ips" {
  description = "IP addresses allowed to SSH"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
```

### outputs.tf
```hcl
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "instance_ids" {
  description = "EC2 instance IDs"
  value       = aws_instance.web[*].id
}

output "instance_public_ips" {
  description = "Public IPs of EC2 instances"
  value       = aws_instance.web[*].public_ip
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.web.id
}
```

## Modules

### Creating a Module

```hcl
# modules/vpc/main.tf
resource "aws_vpc" "this" {
  cidr_block = var.cidr_block

  tags = {
    Name = var.vpc_name
  }
}

resource "aws_subnet" "public" {
  count      = length(var.public_subnet_cidrs)
  vpc_id     = aws_vpc.this.id
  cidr_block = var.public_subnet_cidrs[count.index]

  tags = {
    Name = "${var.vpc_name}-public-${count.index + 1}"
  }
}
```

```hcl
# modules/vpc/variables.tf
variable "vpc_name" {
  type = string
}

variable "cidr_block" {
  type = string
}

variable "public_subnet_cidrs" {
  type = list(string)
}
```

```hcl
# modules/vpc/outputs.tf
output "vpc_id" {
  value = aws_vpc.this.id
}

output "subnet_ids" {
  value = aws_subnet.public[*].id
}
```

### Using a Module

```hcl
# main.tf
module "vpc" {
  source = "./modules/vpc"

  vpc_name             = "production-vpc"
  cidr_block           = "10.0.0.0/16"
  public_subnet_cidrs  = ["10.0.1.0/24", "10.0.2.0/24"]
}

# Reference module outputs
resource "aws_instance" "example" {
  subnet_id = module.vpc.subnet_ids[0]
  # ...
}
```

### Using Public Modules

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"

  name = "my-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = false

  tags = {
    Environment = "production"
  }
}
```

## Backend Configuration

### S3 Backend (AWS)

```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-lock"
  }
}
```

### Azure Backend

```hcl
terraform {
  backend "azurerm" {
    resource_group_name  = "terraform-state-rg"
    storage_account_name = "tfstate"
    container_name       = "tfstate"
    key                  = "prod.terraform.tfstate"
  }
}
```

### Remote Backend (Terraform Cloud)

```hcl
terraform {
  backend "remote" {
    organization = "my-org"

    workspaces {
      name = "production"
    }
  }
}
```

## Advanced Features

### Loops

```hcl
# count
resource "aws_instance" "server" {
  count         = 3
  ami           = "ami-123456"
  instance_type = "t2.micro"

  tags = {
    Name = "server-${count.index}"
  }
}

# for_each with set
resource "aws_iam_user" "users" {
  for_each = toset(["user1", "user2", "user3"])
  name     = each.key
}

# for_each with map
resource "aws_instance" "servers" {
  for_each = {
    web  = "t2.micro"
    app  = "t2.small"
    db   = "t2.medium"
  }

  instance_type = each.value
  ami           = "ami-123456"

  tags = {
    Name = each.key
  }
}
```

### Conditionals

```hcl
# Conditional resource creation
resource "aws_instance" "example" {
  count = var.create_instance ? 1 : 0
  # ...
}

# Conditional expression
resource "aws_instance" "example" {
  instance_type = var.environment == "prod" ? "t2.large" : "t2.micro"
}
```

### Dynamic Blocks

```hcl
resource "aws_security_group" "example" {
  name = "example-sg"

  dynamic "ingress" {
    for_each = var.ingress_rules
    content {
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
    }
  }
}
```

### Local Values

```hcl
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }

  instance_name = "${var.project_name}-${var.environment}-web"
}

resource "aws_instance" "example" {
  tags = merge(
    local.common_tags,
    {
      Name = local.instance_name
    }
  )
}
```

## Best Practices

### 1. Use Version Control
Store Terraform configurations in Git.

### 2. Use Remote State
Store state remotely with locking enabled.

### 3. Use Modules
Create reusable modules for common patterns.

### 4. Use Variables
Don't hardcode values; use variables.

### 5. Use .gitignore
```
# .gitignore
.terraform/
*.tfstate
*.tfstate.*
.terraform.lock.hcl
terraform.tfvars
*.auto.tfvars
```

### 6. Use Terraform Formatting
```bash
terraform fmt -recursive
```

### 7. Pin Provider Versions
```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"  # Pin to major version
    }
  }
}
```

### 8. Use Workspaces
Separate environments with workspaces or separate state files.

### 9. Use Data Sources
Query existing resources instead of hardcoding.

### 10. Plan Before Apply
Always run `terraform plan` before `apply`.

## Common Issues and Solutions

### Issue: State Lock
```bash
# Force unlock (use carefully)
terraform force-unlock <lock-id>
```

### Issue: Drift Detection
```bash
# Refresh state and show differences
terraform plan -refresh-only
```

### Issue: Import Existing Resources
```bash
terraform import aws_instance.example i-1234567890abcdef0
```

### Issue: Debug Mode
```bash
export TF_LOG=DEBUG
terraform apply
```

## Useful Resources

- Official Documentation: https://www.terraform.io/docs
- Terraform Registry: https://registry.terraform.io/
- AWS Provider Docs: https://registry.terraform.io/providers/hashicorp/aws/latest/docs
- Learn Terraform: https://learn.hashicorp.com/terraform
- Style Guide: https://www.terraform.io/docs/language/syntax/style.html

## Quick Reference

| Command | Description |
|---------|-------------|
| `terraform init` | Initialize directory |
| `terraform plan` | Preview changes |
| `terraform apply` | Apply changes |
| `terraform destroy` | Destroy resources |
| `terraform fmt` | Format code |
| `terraform validate` | Validate configuration |
| `terraform state list` | List resources in state |
| `terraform output` | Show outputs |
| `terraform workspace list` | List workspaces |
| `terraform import` | Import existing resource |

---

*This guide covers Terraform fundamentals. For production use, consider implementing CI/CD pipelines, security scanning, cost estimation, and proper state management strategies.*
