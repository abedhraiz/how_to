# AWS (Amazon Web Services) Guide

## What is AWS?

Amazon Web Services (AWS) is the world's most comprehensive and broadly adopted cloud platform, offering over 200 fully featured services from data centers globally. AWS provides on-demand computing resources and services including compute power, storage, databases, networking, and more.

## Prerequisites

- AWS Account (sign up at https://aws.amazon.com/)
- Basic understanding of cloud computing concepts
- Command line knowledge
- Credit card for account verification (free tier available)

## Core Services Overview

### Compute
- **EC2** - Virtual servers in the cloud
- **Lambda** - Serverless compute
- **ECS/EKS** - Container orchestration
- **Elastic Beanstalk** - Platform as a Service

### Storage
- **S3** - Object storage
- **EBS** - Block storage for EC2
- **EFS** - Elastic file system
- **Glacier** - Archive storage

### Database
- **RDS** - Managed relational databases
- **DynamoDB** - NoSQL database
- **ElastiCache** - In-memory cache
- **Redshift** - Data warehouse

### Networking
- **VPC** - Virtual private cloud
- **Route 53** - DNS service
- **CloudFront** - CDN
- **ELB** - Load balancing

## Installation and Setup

### AWS CLI

```bash
# Install AWS CLI (Linux/Mac)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install AWS CLI (Mac with Homebrew)
brew install awscli

# Install AWS CLI (Windows)
# Download from: https://awscli.amazonaws.com/AWSCLIV2.msi

# Verify installation
aws --version
```

### Configure AWS CLI

```bash
# Configure with access keys
aws configure

# You'll be prompted for:
# AWS Access Key ID: YOUR_ACCESS_KEY
# AWS Secret Access Key: YOUR_SECRET_KEY
# Default region name: us-east-1
# Default output format: json

# View configuration
aws configure list

# Use named profiles
aws configure --profile production

# Set default profile
export AWS_PROFILE=production
```

### Get Access Keys

1. Go to IAM Console
2. Click on your username
3. Go to "Security credentials"
4. Click "Create access key"
5. Save the Access Key ID and Secret Access Key

## EC2 (Elastic Compute Cloud)

### Launch an Instance via Console

1. Navigate to EC2 Dashboard
2. Click "Launch Instance"
3. Choose AMI (Amazon Linux, Ubuntu, etc.)
4. Select instance type (t2.micro for free tier)
5. Configure instance details
6. Add storage
7. Add tags
8. Configure security group
9. Review and launch

### Launch Instance via CLI

```bash
# Create key pair
aws ec2 create-key-pair \
    --key-name MyKeyPair \
    --query 'KeyMaterial' \
    --output text > MyKeyPair.pem

chmod 400 MyKeyPair.pem

# Launch instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --count 1 \
    --instance-type t2.micro \
    --key-name MyKeyPair \
    --security-group-ids sg-903004f8 \
    --subnet-id subnet-6e7f829e \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=MyInstance}]'

# List instances
aws ec2 describe-instances

# Get instance status
aws ec2 describe-instance-status --instance-ids i-1234567890abcdef0

# Connect to instance
ssh -i MyKeyPair.pem ec2-user@<public-ip>

# Stop instance
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Start instance
aws ec2 start-instances --instance-ids i-1234567890abcdef0

# Terminate instance
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
```

### EC2 User Data (Bootstrap Script)

```bash
#!/bin/bash
yum update -y
yum install -y httpd
systemctl start httpd
systemctl enable httpd
echo "<h1>Hello from $(hostname -f)</h1>" > /var/www/html/index.html
```

### Security Groups

```bash
# Create security group
aws ec2 create-security-group \
    --group-name MySecurityGroup \
    --description "My security group" \
    --vpc-id vpc-1a2b3c4d

# Add inbound rule (allow SSH)
aws ec2 authorize-security-group-ingress \
    --group-id sg-903004f8 \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

# Add HTTP rule
aws ec2 authorize-security-group-ingress \
    --group-id sg-903004f8 \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

# Remove rule
aws ec2 revoke-security-group-ingress \
    --group-id sg-903004f8 \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

# Delete security group
aws ec2 delete-security-group --group-id sg-903004f8
```

## S3 (Simple Storage Service)

### Basic S3 Operations

```bash
# Create bucket
aws s3 mb s3://my-unique-bucket-name

# List buckets
aws s3 ls

# Upload file
aws s3 cp myfile.txt s3://my-bucket/

# Upload directory
aws s3 cp mydir/ s3://my-bucket/ --recursive

# Download file
aws s3 cp s3://my-bucket/myfile.txt ./

# List bucket contents
aws s3 ls s3://my-bucket/

# Sync local directory to S3
aws s3 sync ./local-folder s3://my-bucket/folder/

# Sync S3 to local
aws s3 sync s3://my-bucket/folder/ ./local-folder/

# Delete file
aws s3 rm s3://my-bucket/myfile.txt

# Delete bucket (must be empty)
aws s3 rb s3://my-bucket

# Delete bucket and all contents
aws s3 rb s3://my-bucket --force
```

### S3 Bucket Policies

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::my-bucket/*"
    }
  ]
}
```

Apply policy:
```bash
aws s3api put-bucket-policy \
    --bucket my-bucket \
    --policy file://policy.json
```

### S3 Lifecycle Rules

```bash
# Create lifecycle configuration
cat > lifecycle.json << EOF
{
  "Rules": [
    {
      "Id": "Move to Glacier",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ],
      "Expiration": {
        "Days": 365
      }
    }
  ]
}
EOF

# Apply lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-bucket \
    --lifecycle-configuration file://lifecycle.json
```

### S3 Versioning

```bash
# Enable versioning
aws s3api put-bucket-versioning \
    --bucket my-bucket \
    --versioning-configuration Status=Enabled

# List versions
aws s3api list-object-versions --bucket my-bucket
```

## Lambda (Serverless Compute)

### Create Lambda Function via CLI

```bash
# Create function
aws lambda create-function \
    --function-name my-function \
    --runtime python3.9 \
    --role arn:aws:iam::123456789012:role/lambda-role \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://function.zip

# Invoke function
aws lambda invoke \
    --function-name my-function \
    --payload '{"key": "value"}' \
    response.json

# Update function code
aws lambda update-function-code \
    --function-name my-function \
    --zip-file fileb://function.zip

# Delete function
aws lambda delete-function --function-name my-function
```

### Python Lambda Example

```python
# lambda_function.py
import json

def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")
    
    # Process event
    name = event.get('name', 'World')
    
    return {
        'statusCode': 200,
        'body': json.dumps(f'Hello, {name}!')
    }
```

### Lambda with S3 Trigger

```python
import json
import boto3

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Get bucket and object key from event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    print(f"New file uploaded: {key} in bucket {bucket}")
    
    # Process the file
    obj = s3.get_object(Bucket=bucket, Key=key)
    content = obj['Body'].read().decode('utf-8')
    
    # Do something with content
    processed = content.upper()
    
    # Save processed file
    s3.put_object(
        Bucket=bucket,
        Key=f"processed/{key}",
        Body=processed
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps('File processed successfully')
    }
```

## RDS (Relational Database Service)

### Create RDS Instance

```bash
# Create PostgreSQL instance
aws rds create-db-instance \
    --db-instance-identifier mydbinstance \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username admin \
    --master-user-password MyPassword123 \
    --allocated-storage 20 \
    --vpc-security-group-ids sg-12345678 \
    --db-subnet-group-name mysubnetgroup

# List instances
aws rds describe-db-instances

# Get endpoint
aws rds describe-db-instances \
    --db-instance-identifier mydbinstance \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text

# Create snapshot
aws rds create-db-snapshot \
    --db-instance-identifier mydbinstance \
    --db-snapshot-identifier mysnapshot

# Restore from snapshot
aws rds restore-db-instance-from-db-snapshot \
    --db-instance-identifier mynewinstance \
    --db-snapshot-identifier mysnapshot

# Delete instance
aws rds delete-db-instance \
    --db-instance-identifier mydbinstance \
    --skip-final-snapshot
```

### Connect to RDS

```bash
# PostgreSQL
psql -h mydbinstance.xxxx.us-east-1.rds.amazonaws.com -U admin -d postgres

# MySQL
mysql -h mydbinstance.xxxx.us-east-1.rds.amazonaws.com -u admin -p
```

## VPC (Virtual Private Cloud)

### Create VPC

```bash
# Create VPC
aws ec2 create-vpc --cidr-block 10.0.0.0/16

# Create subnet
aws ec2 create-subnet \
    --vpc-id vpc-12345678 \
    --cidr-block 10.0.1.0/24 \
    --availability-zone us-east-1a

# Create internet gateway
aws ec2 create-internet-gateway

# Attach gateway to VPC
aws ec2 attach-internet-gateway \
    --vpc-id vpc-12345678 \
    --internet-gateway-id igw-12345678

# Create route table
aws ec2 create-route-table --vpc-id vpc-12345678

# Add route to internet
aws ec2 create-route \
    --route-table-id rtb-12345678 \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id igw-12345678

# Associate route table with subnet
aws ec2 associate-route-table \
    --subnet-id subnet-12345678 \
    --route-table-id rtb-12345678
```

## IAM (Identity and Access Management)

### Create IAM User

```bash
# Create user
aws iam create-user --user-name john

# Create access key
aws iam create-access-key --user-name john

# Attach policy to user
aws iam attach-user-policy \
    --user-name john \
    --policy-arn arn:aws:iam::aws:policy/ReadOnlyAccess

# List users
aws iam list-users

# Delete user
aws iam delete-user --user-name john
```

### Create IAM Role

```bash
# Create trust policy
cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
    --role-name MyEC2Role \
    --assume-role-policy-document file://trust-policy.json

# Attach policy to role
aws iam attach-role-policy \
    --role-name MyEC2Role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

### Create Custom IAM Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::my-bucket/*"
    },
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::my-bucket"
    }
  ]
}
```

```bash
# Create policy
aws iam create-policy \
    --policy-name MyS3Policy \
    --policy-document file://policy.json
```

## CloudFormation (Infrastructure as Code)

### Simple CloudFormation Template

```yaml
# stack.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Simple web server stack

Parameters:
  KeyName:
    Type: AWS::EC2::KeyPair::KeyName
    Description: EC2 Key Pair
  
  InstanceType:
    Type: String
    Default: t2.micro
    AllowedValues:
      - t2.micro
      - t2.small
      - t2.medium

Resources:
  WebServerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Enable HTTP and SSH
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0

  WebServer:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: !Ref InstanceType
      ImageId: ami-0c55b159cbfafe1f0
      KeyName: !Ref KeyName
      SecurityGroups:
        - !Ref WebServerSecurityGroup
      UserData:
        Fn::Base64: |
          #!/bin/bash
          yum update -y
          yum install -y httpd
          systemctl start httpd
          systemctl enable httpd
          echo "<h1>Hello from CloudFormation</h1>" > /var/www/html/index.html

Outputs:
  WebsiteURL:
    Description: URL of the website
    Value: !Sub 'http://${WebServer.PublicDnsName}'
  
  InstanceId:
    Description: Instance ID
    Value: !Ref WebServer
```

### Deploy CloudFormation Stack

```bash
# Create stack
aws cloudformation create-stack \
    --stack-name my-web-server \
    --template-body file://stack.yaml \
    --parameters ParameterKey=KeyName,ParameterValue=MyKeyPair

# List stacks
aws cloudformation list-stacks

# Describe stack
aws cloudformation describe-stacks --stack-name my-web-server

# Update stack
aws cloudformation update-stack \
    --stack-name my-web-server \
    --template-body file://stack.yaml \
    --parameters ParameterKey=KeyName,ParameterValue=MyKeyPair

# Delete stack
aws cloudformation delete-stack --stack-name my-web-server

# View stack events
aws cloudformation describe-stack-events --stack-name my-web-server
```

## ECS (Elastic Container Service)

### Create ECS Cluster

```bash
# Create cluster
aws ecs create-cluster --cluster-name my-cluster

# Register task definition
cat > task-definition.json << EOF
{
  "family": "web-app",
  "containerDefinitions": [
    {
      "name": "web",
      "image": "nginx:latest",
      "memory": 512,
      "cpu": 256,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 80,
          "protocol": "tcp"
        }
      ]
    }
  ]
}
EOF

aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
    --cluster my-cluster \
    --service-name web-service \
    --task-definition web-app \
    --desired-count 2

# List services
aws ecs list-services --cluster my-cluster

# Update service
aws ecs update-service \
    --cluster my-cluster \
    --service web-service \
    --desired-count 3

# Delete service
aws ecs delete-service \
    --cluster my-cluster \
    --service web-service \
    --force
```

## CloudWatch (Monitoring & Logging)

### CloudWatch Logs

```bash
# Create log group
aws logs create-log-group --log-group-name /aws/lambda/my-function

# Create log stream
aws logs create-log-stream \
    --log-group-name /aws/lambda/my-function \
    --log-stream-name 2024/01/01

# Put log events
aws logs put-log-events \
    --log-group-name /aws/lambda/my-function \
    --log-stream-name 2024/01/01 \
    --log-events timestamp=1609459200000,message="Log message"

# Get logs
aws logs filter-log-events \
    --log-group-name /aws/lambda/my-function \
    --start-time 1609459200000

# Tail logs
aws logs tail /aws/lambda/my-function --follow
```

### CloudWatch Alarms

```bash
# Create alarm
aws cloudwatch put-metric-alarm \
    --alarm-name high-cpu \
    --alarm-description "Alert when CPU exceeds 80%" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --dimensions Name=InstanceId,Value=i-1234567890abcdef0

# List alarms
aws cloudwatch describe-alarms

# Delete alarm
aws cloudwatch delete-alarms --alarm-names high-cpu
```

## Route 53 (DNS Service)

### Create Hosted Zone

```bash
# Create hosted zone
aws route53 create-hosted-zone \
    --name example.com \
    --caller-reference $(date +%s)

# List hosted zones
aws route53 list-hosted-zones

# Create record set
cat > change-batch.json << EOF
{
  "Changes": [
    {
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "www.example.com",
        "Type": "A",
        "TTL": 300,
        "ResourceRecords": [
          {
            "Value": "192.0.2.1"
          }
        ]
      }
    }
  ]
}
EOF

aws route53 change-resource-record-sets \
    --hosted-zone-id Z123456789ABC \
    --change-batch file://change-batch.json
```

## AWS SDK (Boto3 for Python)

### Installation

```bash
pip install boto3
```

### Basic Usage

```python
import boto3

# Create EC2 client
ec2 = boto3.client('ec2', region_name='us-east-1')

# List instances
response = ec2.describe_instances()
for reservation in response['Reservations']:
    for instance in reservation['Instances']:
        print(f"Instance ID: {instance['InstanceId']}")
        print(f"State: {instance['State']['Name']}")

# Create S3 client
s3 = boto3.client('s3')

# Upload file
s3.upload_file('local-file.txt', 'my-bucket', 'remote-file.txt')

# Download file
s3.download_file('my-bucket', 'remote-file.txt', 'local-file.txt')

# List buckets
response = s3.list_buckets()
for bucket in response['Buckets']:
    print(f"Bucket: {bucket['Name']}")
```

### DynamoDB Operations

```python
import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

# Create table
table = dynamodb.create_table(
    TableName='Users',
    KeySchema=[
        {'AttributeName': 'user_id', 'KeyType': 'HASH'}
    ],
    AttributeDefinitions=[
        {'AttributeName': 'user_id', 'AttributeType': 'S'}
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

# Wait for table to be created
table.wait_until_exists()

# Put item
table = dynamodb.Table('Users')
table.put_item(
    Item={
        'user_id': '123',
        'name': 'John Doe',
        'email': 'john@example.com'
    }
)

# Get item
response = table.get_item(Key={'user_id': '123'})
item = response['Item']
print(item)

# Query items
response = table.scan()
items = response['Items']
for item in items:
    print(item)

# Update item
table.update_item(
    Key={'user_id': '123'},
    UpdateExpression='SET email = :email',
    ExpressionAttributeValues={':email': 'newemail@example.com'}
)

# Delete item
table.delete_item(Key={'user_id': '123'})
```

## Cost Optimization

### Best Practices

1. **Use Free Tier** - 750 hours/month of t2.micro, 5GB S3 storage
2. **Right-Size Instances** - Use appropriate instance types
3. **Use Reserved Instances** - Up to 75% savings for predictable workloads
4. **Use Spot Instances** - Up to 90% savings for flexible workloads
5. **Auto-Scaling** - Scale resources based on demand
6. **S3 Lifecycle Policies** - Move old data to cheaper storage
7. **Delete Unused Resources** - Remove idle instances, snapshots
8. **Use CloudWatch** - Monitor and optimize resource usage

### Cost Explorer

```bash
# Get cost and usage
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity MONTHLY \
    --metrics "BlendedCost" "UnblendedCost"
```

### Set up Billing Alerts

```bash
# Create SNS topic
aws sns create-topic --name billing-alerts

# Subscribe to topic
aws sns subscribe \
    --topic-arn arn:aws:sns:us-east-1:123456789012:billing-alerts \
    --protocol email \
    --notification-endpoint your-email@example.com

# Create billing alarm
aws cloudwatch put-metric-alarm \
    --alarm-name billing-alarm \
    --alarm-description "Alert when bill exceeds $100" \
    --metric-name EstimatedCharges \
    --namespace AWS/Billing \
    --statistic Maximum \
    --period 21600 \
    --evaluation-periods 1 \
    --threshold 100 \
    --comparison-operator GreaterThanThreshold
```

## Security Best Practices

1. **Enable MFA** - Multi-factor authentication for root and IAM users
2. **Use IAM Roles** - Don't use root account or embed access keys
3. **Principle of Least Privilege** - Grant minimum required permissions
4. **Enable CloudTrail** - Log all API calls
5. **Encrypt Data** - Use encryption at rest and in transit
6. **Security Groups** - Restrict inbound/outbound traffic
7. **Regular Audits** - Use AWS Config and Security Hub
8. **Patch Management** - Keep instances updated
9. **VPC** - Use private subnets for sensitive resources
10. **Backup Data** - Regular snapshots and backups

## Common AWS CLI Commands

```bash
# EC2
aws ec2 describe-instances
aws ec2 start-instances --instance-ids i-xxx
aws ec2 stop-instances --instance-ids i-xxx
aws ec2 terminate-instances --instance-ids i-xxx

# S3
aws s3 ls
aws s3 cp file.txt s3://bucket/
aws s3 sync ./folder s3://bucket/folder/
aws s3 rm s3://bucket/file.txt

# Lambda
aws lambda list-functions
aws lambda invoke --function-name my-function output.json

# RDS
aws rds describe-db-instances
aws rds create-db-snapshot --db-instance-identifier mydb --db-snapshot-identifier snapshot1

# IAM
aws iam list-users
aws iam create-user --user-name john
aws iam list-roles

# CloudFormation
aws cloudformation list-stacks
aws cloudformation describe-stacks --stack-name my-stack
aws cloudformation delete-stack --stack-name my-stack

# CloudWatch
aws logs tail /aws/lambda/function-name --follow
aws cloudwatch describe-alarms
```

## Troubleshooting

### Common Issues

**Issue: Access Denied**
```bash
# Check IAM permissions
aws iam get-user
aws sts get-caller-identity

# Verify credentials
aws configure list
```

**Issue: Instance Not Accessible**
```bash
# Check security group rules
aws ec2 describe-security-groups --group-ids sg-xxx

# Check instance status
aws ec2 describe-instance-status --instance-ids i-xxx

# Check system log
aws ec2 get-console-output --instance-id i-xxx
```

**Issue: S3 Access Denied**
```bash
# Check bucket policy
aws s3api get-bucket-policy --bucket my-bucket

# Check bucket ACL
aws s3api get-bucket-acl --bucket my-bucket
```

## Resources

- **AWS Documentation**: https://docs.aws.amazon.com/
- **AWS CLI Reference**: https://awscli.amazonaws.com/v2/documentation/api/latest/index.html
- **AWS Free Tier**: https://aws.amazon.com/free/
- **AWS Architecture Center**: https://aws.amazon.com/architecture/
- **AWS Well-Architected**: https://aws.amazon.com/architecture/well-architected/
- **AWS Training**: https://aws.amazon.com/training/

## Quick Reference

| Service | Purpose | Common Commands |
|---------|---------|-----------------|
| EC2 | Virtual servers | `aws ec2 describe-instances` |
| S3 | Object storage | `aws s3 ls`, `aws s3 cp` |
| Lambda | Serverless compute | `aws lambda invoke` |
| RDS | Managed databases | `aws rds describe-db-instances` |
| VPC | Networking | `aws ec2 describe-vpcs` |
| IAM | Access management | `aws iam list-users` |
| CloudFormation | Infrastructure as Code | `aws cloudformation create-stack` |
| ECS | Container orchestration | `aws ecs list-clusters` |
| CloudWatch | Monitoring | `aws logs tail` |
| Route 53 | DNS | `aws route53 list-hosted-zones` |

---

*This guide covers AWS fundamentals and most commonly used services. AWS offers 200+ services - explore the official documentation for advanced features and specialized services.*
