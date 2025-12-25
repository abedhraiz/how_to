# n8n Workflow Automation Guide

## What is n8n?

n8n (pronounced "n-eight-n") is a free and open-source workflow automation tool that allows you to connect various apps and services together. It's a powerful alternative to tools like Zapier, Integromat, or Make, with the advantage of being self-hostable.

## Key Features

- **Open Source** - Free to use and modify
- **Self-Hosted** - Full control over your data
- **Visual Workflow Builder** - Drag-and-drop interface
- **400+ Integrations** - Connect to popular services
- **Custom Code** - JavaScript/Python for complex logic
- **API Access** - REST API for automation
- **Fair-Code License** - Free for self-hosting

## Installation

### Using npm (Recommended for Quick Start)

```bash
# Install globally
npm install n8n -g

# Start n8n
n8n start

# Access at http://localhost:5678
```

### Using Docker

```bash
# Run n8n container
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n

# With persistent data
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -e N8N_BASIC_AUTH_ACTIVE=true \
  -e N8N_BASIC_AUTH_USER=admin \
  -e N8N_BASIC_AUTH_PASSWORD=password \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n
```

### Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  n8n:
    image: n8nio/n8n
    restart: always
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=password
      - N8N_HOST=localhost
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - NODE_ENV=production
      - WEBHOOK_URL=http://localhost:5678/
    volumes:
      - ~/.n8n:/home/node/.n8n
```

Run with:
```bash
docker-compose up -d
```

## Core Concepts

### 1. **Workflows**
A series of connected nodes that automate a process.

### 2. **Nodes**
Individual steps in a workflow. Each node performs a specific action.

Types of nodes:
- **Trigger Nodes** - Start the workflow (webhooks, schedules, etc.)
- **Regular Nodes** - Perform actions (send email, create record, etc.)
- **Core Nodes** - Built-in functionality (if, merge, split, etc.)

### 3. **Connections**
Lines that connect nodes, defining the flow of data.

### 4. **Executions**
Individual runs of a workflow with their data and status.

### 5. **Credentials**
Stored authentication information for services.

## Building Your First Workflow

### Step 1: Create a New Workflow

1. Click "+" or "New Workflow"
2. Give your workflow a name

### Step 2: Add a Trigger Node

Common triggers:
- **Webhook** - HTTP requests
- **Schedule Trigger** - Cron-based scheduling
- **Manual Trigger** - Run manually
- **Email Trigger** - On new email
- **Form Trigger** - Web form submissions

Example: Schedule Trigger
```
Node: Schedule Trigger
Settings:
  - Mode: Every day
  - Hour: 9
  - Minute: 0
```

### Step 3: Add Action Nodes

Click "+" to add nodes and connect them to your trigger.

### Step 4: Test Your Workflow

Click "Execute Workflow" to test.

### Step 5: Activate

Toggle the switch at the top to activate your workflow.

## Common Workflow Patterns

### 1. Webhook to Database

```
Webhook → HTTP Request → Set → Database (Insert)
```

**Use Case:** Receive data via webhook and save to database

### 2. Scheduled Data Sync

```
Schedule Trigger → API Request → Transform Data → Google Sheets
```

**Use Case:** Daily data synchronization

### 3. Form Submission Handler

```
Webhook → Split In Batches → Send Email → Slack Notification
```

**Use Case:** Process form submissions and notify team

### 4. Email Automation

```
Email Trigger (IMAP) → If → Gmail (Send) → Trello (Create Card)
```

**Use Case:** Auto-respond to emails and create tasks

### 5. Data Processing Pipeline

```
HTTP Request → Function → Filter → Split → Multiple Actions
```

**Use Case:** Fetch, process, and distribute data

## Essential Nodes

### Trigger Nodes

#### Webhook
```
Trigger: Webhook
Method: POST
Path: my-webhook
Authentication: None
```

#### Schedule (Cron)
```
Trigger: Schedule Trigger
Mode: Custom (Cron)
Cron Expression: 0 9 * * 1-5  # 9 AM weekdays
```

### Core Nodes

#### Function Node (JavaScript)
```javascript
// Access input data
const items = $input.all();

// Process data
for (let item of items) {
  item.json.newField = item.json.oldField * 2;
}

// Return modified data
return items;
```

#### Code Node (Python)
```python
# Access input items
items = _input.all()

# Process data
for item in items:
    item.json['processed'] = True
    item.json['timestamp'] = datetime.now().isoformat()

# Return items
return items
```

#### IF Node
```
Condition:
  - Field: {{ $json.status }}
  - Operation: Equal
  - Value: success
```

#### Set Node
```
Add fields:
  - name: fullName
    value: {{ $json.firstName }} {{ $json.lastName }}
  - name: timestamp
    value: {{ $now.toISO() }}
```

#### HTTP Request
```
Method: POST
URL: https://api.example.com/data
Authentication: Header Auth
Headers:
  - Authorization: Bearer {{ $credentials.apiToken }}
Body:
  {
    "data": "{{ $json.data }}"
  }
```

### Integration Nodes

#### Gmail
```
Operation: Send Email
To: user@example.com
Subject: {{ $json.subject }}
Body: {{ $json.message }}
```

#### Slack
```
Operation: Post Message
Channel: #general
Text: New submission from {{ $json.name }}
```

#### Google Sheets
```
Operation: Append
Sheet: Sheet1
Columns:
  - Name: {{ $json.name }}
  - Email: {{ $json.email }}
  - Date: {{ $now.toFormat('yyyy-MM-dd') }}
```

#### MySQL
```
Operation: Execute Query
Query: INSERT INTO users (name, email) VALUES (?, ?)
Parameters: {{ $json.name }}, {{ $json.email }}
```

## Expressions and Variables

### Accessing Data

```javascript
// Current item
{{ $json.fieldName }}

// All items
{{ $items }}

// Input data
{{ $input.first().json.field }}

// Previous node data
{{ $node["Node Name"].json.field }}

// Item index
{{ $itemIndex }}
```

### Date and Time

```javascript
// Current date/time
{{ $now }}

// Format date
{{ $now.toFormat('yyyy-MM-dd') }}

// Add days
{{ $now.plus({ days: 7 }).toISO() }}

// Parse date
{{ DateTime.fromISO($json.date) }}
```

### String Operations

```javascript
// Concatenation
{{ $json.firstName + ' ' + $json.lastName }}

// Uppercase
{{ $json.text.toUpperCase() }}

// Substring
{{ $json.text.substring(0, 10) }}

// Replace
{{ $json.text.replace('old', 'new') }}
```

### Array Operations

```javascript
// Length
{{ $json.items.length }}

// Join
{{ $json.tags.join(', ') }}

// Filter
{{ $json.items.filter(item => item.active) }}

// Map
{{ $json.items.map(item => item.name) }}
```

### Conditional Logic

```javascript
// Ternary
{{ $json.status === 'active' ? 'Yes' : 'No' }}

// Nullish coalescing
{{ $json.value ?? 'default' }}
```

## Working with Credentials

### Add Credentials

1. Click on a node requiring authentication
2. Select "Create New Credential"
3. Enter required information
4. Save

### Credential Types

- **OAuth2** - Google, GitHub, etc.
- **API Key** - Simple token authentication
- **Header Auth** - Custom headers
- **Basic Auth** - Username/password
- **JWT** - JSON Web Tokens

### Using Credentials in Code

```javascript
// Access credential data
const apiKey = $credentials.apiKey;
const token = $credentials.oAuth2.accessToken;
```

## Error Handling

### Error Trigger Node

```
Trigger: Error Trigger
Workflow: Select workflow to monitor
```

### Try-Catch Pattern

```
Main Flow → [Error Trigger] → Notification
```

### IF Node for Validation

```javascript
// Check if required field exists
{{ $json.email !== undefined && $json.email !== '' }}
```

### Function Node Error Handling

```javascript
try {
  // Your code here
  const result = processData($json.data);
  return [{ json: { result } }];
} catch (error) {
  return [{ json: { error: error.message } }];
}
```

## Advanced Features

### Split In Batches

Process large datasets in chunks:

```
HTTP Request → Split In Batches → Process → Loop Back
Settings:
  - Batch Size: 100
```

### Merge Node

Combine data from multiple branches:

```
Branch A ↘
          Merge → Continue
Branch B ↗

Mode: Merge By Position / Merge By Key
```

### Wait Node

Add delays between actions:

```
Settings:
  - Wait Time: 5 seconds / minutes / hours
```

### Sticky Note

Add comments to document workflows:

```
Double-click canvas → Add Sticky Note
```

## Webhooks

### Creating a Webhook

1. Add "Webhook" node
2. Choose HTTP Method (GET, POST, etc.)
3. Set path (e.g., `/my-webhook`)
4. Copy the webhook URL
5. Activate workflow

### Webhook URL Format

```
http://localhost:5678/webhook/my-webhook

# Production
https://your-domain.com/webhook/my-webhook

# Test URL (while editing)
http://localhost:5678/webhook-test/my-webhook
```

### Testing Webhooks

```bash
# Using curl
curl -X POST http://localhost:5678/webhook/my-webhook \
  -H "Content-Type: application/json" \
  -d '{"name": "John", "email": "john@example.com"}'
```

## Environment Variables

### Common Variables

```bash
# Basic Auth
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=password

# Host Configuration
N8N_HOST=0.0.0.0
N8N_PORT=5678
N8N_PROTOCOL=https

# Webhook URL
WEBHOOK_URL=https://your-domain.com/

# Database
DB_TYPE=postgresdb
DB_POSTGRESDB_HOST=localhost
DB_POSTGRESDB_PORT=5432
DB_POSTGRESDB_DATABASE=n8n
DB_POSTGRESDB_USER=n8n
DB_POSTGRESDB_PASSWORD=password

# Execution
EXECUTIONS_DATA_SAVE_ON_ERROR=all
EXECUTIONS_DATA_SAVE_ON_SUCCESS=all
EXECUTIONS_DATA_PRUNE=true
EXECUTIONS_DATA_MAX_AGE=336  # hours

# Timezone
GENERIC_TIMEZONE=America/New_York
```

## n8n CLI Commands

```bash
# Start n8n
n8n start

# Import workflows
n8n import:workflow --input=workflow.json

# Export workflows
n8n export:workflow --id=1 --output=workflow.json

# Export all workflows
n8n export:workflow --all --output=./workflows/

# Import credentials
n8n import:credentials --input=credentials.json

# Update n8n
npm update -g n8n
```

## API Usage

### Authentication

```bash
# Get API key from Settings → API
export N8N_API_KEY="your-api-key"
```

### List Workflows

```bash
curl -X GET http://localhost:5678/api/v1/workflows \
  -H "X-N8N-API-KEY: $N8N_API_KEY"
```

### Execute Workflow

```bash
curl -X POST http://localhost:5678/api/v1/workflows/1/execute \
  -H "X-N8N-API-KEY: $N8N_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"data": "test"}'
```

### Get Execution

```bash
curl -X GET http://localhost:5678/api/v1/executions/123 \
  -H "X-N8N-API-KEY: $N8N_API_KEY"
```

## Best Practices

### 1. **Workflow Organization**
- Use descriptive names
- Add sticky notes for documentation
- Group related workflows

### 2. **Error Handling**
- Always add error triggers for critical workflows
- Use IF nodes to validate data
- Log errors to monitoring services

### 3. **Performance**
- Use "Split In Batches" for large datasets
- Limit execution history retention
- Optimize HTTP requests (batch when possible)

### 4. **Security**
- Enable basic auth or use a reverse proxy
- Use HTTPS in production
- Store sensitive data in credentials
- Validate webhook inputs

### 5. **Testing**
- Test workflows before activating
- Use Manual Trigger for testing
- Check execution logs regularly

### 6. **Maintenance**
- Regularly backup workflows
- Keep n8n updated
- Monitor execution history
- Clean up old executions

## Troubleshooting

### Workflow Not Triggering

```
✓ Check if workflow is activated (toggle switch)
✓ Verify trigger node configuration
✓ Check execution logs
✓ Test with Manual Trigger
```

### Webhook Issues

```
✓ Ensure workflow is active
✓ Use production webhook URL
✓ Check authentication settings
✓ Verify HTTP method matches
```

### Node Execution Errors

```
✓ Check credential configuration
✓ Review error message in execution log
✓ Test API endpoint independently
✓ Verify data format
```

### Performance Issues

```
✓ Reduce execution history retention
✓ Use Split In Batches for large data
✓ Optimize database queries
✓ Consider upgrading server resources
```

## Popular Integrations

### Productivity
- Google Workspace (Gmail, Sheets, Drive, Calendar)
- Microsoft 365
- Notion
- Airtable
- Trello, Asana, Jira

### Communication
- Slack
- Discord
- Telegram
- WhatsApp Business
- Email (IMAP/SMTP)

### Development
- GitHub
- GitLab
- Jira
- Linear
- PagerDuty

### Marketing
- Mailchimp
- HubSpot
- SendGrid
- ActiveCampaign
- Social Media APIs

### Databases
- PostgreSQL
- MySQL
- MongoDB
- Redis
- Supabase

### AI/ML
- OpenAI (ChatGPT)
- Anthropic (Claude)
- Hugging Face
- Google AI
- Custom ML APIs

## Community Resources

- **Official Website:** https://n8n.io
- **Documentation:** https://docs.n8n.io
- **Community Forum:** https://community.n8n.io
- **GitHub:** https://github.com/n8n-io/n8n
- **Template Library:** https://n8n.io/workflows
- **YouTube Channel:** n8n.io tutorials
- **Discord:** Community support

## Quick Reference

| Feature | Action |
|---------|--------|
| New Workflow | Click "+" button |
| Add Node | Click "+" on canvas or connection |
| Execute Workflow | Click "Execute Workflow" button |
| Activate Workflow | Toggle switch (top right) |
| View Executions | Click "Executions" tab |
| Add Credentials | Node settings → Credentials |
| Test Webhook | Use "test" URL while editing |
| Copy Workflow | Settings → Duplicate |
| Export Workflow | Settings → Download |
| Import Workflow | Click "⋮" → Import |

## Example: Complete Automation

### Scenario: Customer Onboarding

```
1. Webhook (Form Submission)
   ↓
2. Set (Format Data)
   ↓
3. MySQL (Insert Customer)
   ↓
4. Gmail (Welcome Email)
   ↓
5. Slack (Notify Team)
   ↓
6. Airtable (Add to CRM)
   ↓
7. HTTP Request (Trigger Other Service)
```

---

*This guide covers n8n fundamentals. Explore the template library and community forum for more advanced workflows and integrations.*
