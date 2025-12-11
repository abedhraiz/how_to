# Ansible Guide

## What is Ansible?

Ansible is an open-source automation tool for configuration management, application deployment, and task automation. It uses SSH for communication and doesn't require agents on managed nodes, making it simple and powerful.

## Prerequisites

- Python 3.8+ installed
- SSH access to managed nodes
- Basic understanding of YAML
- Linux/Unix command line knowledge

## Installation

### Linux (Ubuntu/Debian)

```bash
# Update package manager
sudo apt update

# Install Ansible
sudo apt install ansible -y

# Verify installation
ansible --version
```

### Linux (RHEL/CentOS)

```bash
# Install EPEL repository
sudo yum install epel-release -y

# Install Ansible
sudo yum install ansible -y
```

### macOS

```bash
# Using Homebrew
brew install ansible

# Verify
ansible --version
```

### Using pip (All platforms)

```bash
# Install via pip
pip install ansible

# Or for specific version
pip install ansible==9.0.0

# Verify
ansible --version
```

## Core Concepts

### 1. **Control Node**
The machine where Ansible is installed and runs from.

### 2. **Managed Nodes**
Target servers/devices that Ansible manages.

### 3. **Inventory**
List of managed nodes organized into groups.

### 4. **Playbooks**
YAML files that define automation tasks.

### 5. **Modules**
Units of code that Ansible executes (copy, yum, service, etc.).

### 6. **Tasks**
Individual actions executed by modules.

### 7. **Roles**
Organized way to reuse playbooks and related files.

### 8. **Facts**
System information gathered from managed nodes.

## Inventory

### Basic Inventory File

```ini
# inventory.ini or hosts

# Single host
webserver1.example.com

# Hosts with custom SSH port
webserver2.example.com:2222

# Group of hosts
[webservers]
web1.example.com
web2.example.com
web3.example.com

[databases]
db1.example.com
db2.example.com

# Group with variables
[webservers:vars]
http_port=80
max_clients=200

# Nested groups
[production:children]
webservers
databases

[production:vars]
env=production

# Localhost
[local]
localhost ansible_connection=local
```

### Inventory with Variables

```ini
# Host-specific variables
web1.example.com ansible_user=admin ansible_port=2222

# Using IP addresses
[webservers]
192.168.1.10 ansible_user=ubuntu
192.168.1.11 ansible_user=ubuntu

# Patterns
[webservers]
web[01:50].example.com
```

### YAML Inventory

```yaml
# inventory.yml
all:
  hosts:
    web1.example.com:
      ansible_user: admin
  children:
    webservers:
      hosts:
        web2.example.com:
        web3.example.com:
      vars:
        http_port: 80
    databases:
      hosts:
        db1.example.com:
        db2.example.com:
```

## Ad-Hoc Commands

```bash
# Ping all hosts
ansible all -m ping -i inventory.ini

# Ping specific group
ansible webservers -m ping -i inventory.ini

# Check disk space
ansible all -m command -a "df -h"

# Shorter version
ansible all -a "df -h"

# Copy file
ansible webservers -m copy -a "src=/local/file dest=/remote/path"

# Install package (requires sudo)
ansible webservers -m apt -a "name=nginx state=present" --become

# Start service
ansible webservers -m service -a "name=nginx state=started" --become

# Gather facts
ansible all -m setup

# Gather specific facts
ansible all -m setup -a "filter=ansible_distribution*"

# Run as different user
ansible webservers -a "whoami" --become --become-user=www-data

# Set environment variable
ansible all -m shell -a "echo $HOME" -e "HOME=/tmp"
```

## Playbooks

### Basic Playbook Structure

```yaml
# playbook.yml
---
- name: Configure web servers
  hosts: webservers
  become: yes
  
  tasks:
    - name: Install nginx
      apt:
        name: nginx
        state: present
        update_cache: yes
    
    - name: Start nginx service
      service:
        name: nginx
        state: started
        enabled: yes
    
    - name: Copy website files
      copy:
        src: ./website/
        dest: /var/www/html/
        owner: www-data
        group: www-data
        mode: '0644'
```

### Running Playbooks

```bash
# Run playbook
ansible-playbook playbook.yml -i inventory.ini

# Check syntax
ansible-playbook playbook.yml --syntax-check

# Dry run (check mode)
ansible-playbook playbook.yml --check

# Verbose output
ansible-playbook playbook.yml -v
ansible-playbook playbook.yml -vv
ansible-playbook playbook.yml -vvv

# Limit to specific hosts
ansible-playbook playbook.yml --limit webserver1

# Start at specific task
ansible-playbook playbook.yml --start-at-task="Install nginx"

# List tasks
ansible-playbook playbook.yml --list-tasks

# List hosts
ansible-playbook playbook.yml --list-hosts
```

## Playbook Examples

### Complete Web Server Setup

```yaml
---
- name: Setup NGINX web server
  hosts: webservers
  become: yes
  
  vars:
    nginx_port: 80
    document_root: /var/www/html
    server_name: example.com
  
  tasks:
    - name: Update apt cache
      apt:
        update_cache: yes
        cache_valid_time: 3600
    
    - name: Install nginx
      apt:
        name: nginx
        state: present
    
    - name: Create document root
      file:
        path: "{{ document_root }}"
        state: directory
        owner: www-data
        group: www-data
        mode: '0755'
    
    - name: Copy nginx configuration
      template:
        src: templates/nginx.conf.j2
        dest: /etc/nginx/sites-available/default
      notify: Restart nginx
    
    - name: Copy website files
      copy:
        src: website/
        dest: "{{ document_root }}"
        owner: www-data
        group: www-data
    
    - name: Ensure nginx is running
      service:
        name: nginx
        state: started
        enabled: yes
    
    - name: Open firewall for HTTP
      ufw:
        rule: allow
        port: "{{ nginx_port }}"
        proto: tcp
  
  handlers:
    - name: Restart nginx
      service:
        name: nginx
        state: restarted
```

### Database Setup

```yaml
---
- name: Setup MySQL database
  hosts: databases
  become: yes
  
  vars:
    mysql_root_password: "{{ vault_mysql_root_password }}"
    mysql_database: myapp
    mysql_user: appuser
    mysql_password: "{{ vault_mysql_password }}"
  
  tasks:
    - name: Install MySQL
      apt:
        name:
          - mysql-server
          - python3-pymysql
        state: present
    
    - name: Start MySQL service
      service:
        name: mysql
        state: started
        enabled: yes
    
    - name: Set root password
      mysql_user:
        name: root
        password: "{{ mysql_root_password }}"
        login_unix_socket: /var/run/mysqld/mysqld.sock
    
    - name: Create database
      mysql_db:
        name: "{{ mysql_database }}"
        state: present
        login_user: root
        login_password: "{{ mysql_root_password }}"
    
    - name: Create database user
      mysql_user:
        name: "{{ mysql_user }}"
        password: "{{ mysql_password }}"
        priv: "{{ mysql_database }}.*:ALL"
        state: present
        login_user: root
        login_password: "{{ mysql_root_password }}"
```

### Docker Deployment

```yaml
---
- name: Deploy application with Docker
  hosts: app_servers
  become: yes
  
  vars:
    app_name: myapp
    app_version: "1.0.0"
    container_port: 3000
    host_port: 80
  
  tasks:
    - name: Install Docker dependencies
      apt:
        name:
          - apt-transport-https
          - ca-certificates
          - curl
          - software-properties-common
        state: present
    
    - name: Add Docker GPG key
      apt_key:
        url: https://download.docker.com/linux/ubuntu/gpg
        state: present
    
    - name: Add Docker repository
      apt_repository:
        repo: deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable
        state: present
    
    - name: Install Docker
      apt:
        name: docker-ce
        state: present
        update_cache: yes
    
    - name: Install Docker Python library
      pip:
        name: docker
        state: present
    
    - name: Pull application image
      docker_image:
        name: "{{ app_name }}"
        tag: "{{ app_version }}"
        source: pull
    
    - name: Stop existing container
      docker_container:
        name: "{{ app_name }}"
        state: absent
    
    - name: Start new container
      docker_container:
        name: "{{ app_name }}"
        image: "{{ app_name }}:{{ app_version }}"
        state: started
        restart_policy: always
        ports:
          - "{{ host_port }}:{{ container_port }}"
        env:
          NODE_ENV: production
```

## Variables

### Defining Variables

```yaml
---
- name: Variables example
  hosts: webservers
  
  vars:
    # Simple variables
    http_port: 80
    server_name: example.com
    
    # List variables
    packages:
      - nginx
      - git
      - curl
    
    # Dictionary variables
    nginx_config:
      port: 80
      root: /var/www/html
      index: index.html
  
  tasks:
    - name: Use simple variable
      debug:
        msg: "Server name is {{ server_name }}"
    
    - name: Use list variable
      apt:
        name: "{{ packages }}"
        state: present
    
    - name: Use dictionary variable
      debug:
        msg: "Nginx port is {{ nginx_config.port }}"
```

### Variable Files

```yaml
# vars/main.yml
---
app_name: myapp
app_version: 1.0.0
app_port: 3000

database:
  host: localhost
  name: myapp_db
  user: appuser
```

```yaml
# playbook.yml
---
- name: Use variable file
  hosts: all
  vars_files:
    - vars/main.yml
  
  tasks:
    - name: Display app name
      debug:
        msg: "Application: {{ app_name }}"
```

### Host and Group Variables

```bash
# Directory structure
inventory/
  group_vars/
    all.yml
    webservers.yml
    databases.yml
  host_vars/
    web1.example.com.yml
    db1.example.com.yml
```

```yaml
# group_vars/webservers.yml
---
http_port: 80
max_connections: 100
```

```yaml
# host_vars/web1.example.com.yml
---
ansible_user: admin
ansible_port: 2222
```

## Conditionals

```yaml
---
- name: Conditional tasks
  hosts: all
  
  tasks:
    - name: Install Apache (Debian)
      apt:
        name: apache2
        state: present
      when: ansible_os_family == "Debian"
    
    - name: Install Apache (RedHat)
      yum:
        name: httpd
        state: present
      when: ansible_os_family == "RedHat"
    
    - name: Multiple conditions (AND)
      debug:
        msg: "This is production"
      when:
        - env == "production"
        - ansible_distribution == "Ubuntu"
    
    - name: Multiple conditions (OR)
      debug:
        msg: "Dev or staging"
      when: env == "dev" or env == "staging"
    
    - name: Check if variable is defined
      debug:
        msg: "Variable is defined"
      when: my_var is defined
    
    - name: Check if file exists
      debug:
        msg: "File exists"
      when: ansible_facts['stat']['exists']
```

## Loops

```yaml
---
- name: Loop examples
  hosts: all
  
  tasks:
    - name: Create multiple users
      user:
        name: "{{ item }}"
        state: present
      loop:
        - alice
        - bob
        - charlie
    
    - name: Install multiple packages
      apt:
        name: "{{ item }}"
        state: present
      loop:
        - nginx
        - git
        - curl
    
    - name: Loop with dictionary
      user:
        name: "{{ item.name }}"
        uid: "{{ item.uid }}"
        state: present
      loop:
        - { name: 'alice', uid: 1001 }
        - { name: 'bob', uid: 1002 }
        - { name: 'charlie', uid: 1003 }
    
    - name: Loop with variable
      debug:
        msg: "{{ item }}"
      loop: "{{ packages }}"
      vars:
        packages:
          - nginx
          - mysql
    
    - name: Loop until condition
      shell: /usr/bin/check_service
      register: result
      until: result.stdout.find("running") != -1
      retries: 5
      delay: 10
```

## Handlers

```yaml
---
- name: Handlers example
  hosts: webservers
  become: yes
  
  tasks:
    - name: Copy nginx config
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/nginx.conf
      notify:
        - Restart nginx
        - Check nginx status
    
    - name: Copy website files
      copy:
        src: website/
        dest: /var/www/html/
      notify: Reload nginx
  
  handlers:
    - name: Restart nginx
      service:
        name: nginx
        state: restarted
    
    - name: Reload nginx
      service:
        name: nginx
        state: reloaded
    
    - name: Check nginx status
      command: nginx -t
```

## Templates (Jinja2)

### Template File

```jinja2
{# templates/nginx.conf.j2 #}
server {
    listen {{ nginx_port }};
    server_name {{ server_name }};
    
    root {{ document_root }};
    index index.html index.htm;
    
    location / {
        try_files $uri $uri/ =404;
    }
    
    {% if enable_ssl %}
    ssl_certificate {{ ssl_cert_path }};
    ssl_certificate_key {{ ssl_key_path }};
    {% endif %}
    
    {% for location in custom_locations %}
    location {{ location.path }} {
        proxy_pass {{ location.proxy_pass }};
    }
    {% endfor %}
}
```

### Using Template

```yaml
---
- name: Deploy nginx config
  hosts: webservers
  become: yes
  
  vars:
    nginx_port: 80
    server_name: example.com
    document_root: /var/www/html
    enable_ssl: true
    ssl_cert_path: /etc/ssl/certs/server.crt
    ssl_key_path: /etc/ssl/private/server.key
    custom_locations:
      - path: /api
        proxy_pass: http://localhost:3000
      - path: /admin
        proxy_pass: http://localhost:4000
  
  tasks:
    - name: Deploy nginx configuration
      template:
        src: templates/nginx.conf.j2
        dest: /etc/nginx/sites-available/default
      notify: Restart nginx
  
  handlers:
    - name: Restart nginx
      service:
        name: nginx
        state: restarted
```

## Roles

### Role Structure

```
roles/
  webserver/
    tasks/
      main.yml
    handlers/
      main.yml
    templates/
      nginx.conf.j2
    files/
      index.html
    vars/
      main.yml
    defaults/
      main.yml
    meta/
      main.yml
```

### Creating a Role

```bash
# Create role structure
ansible-galaxy init webserver

# Directory created:
# roles/webserver/
```

### Role Files

```yaml
# roles/webserver/tasks/main.yml
---
- name: Install nginx
  apt:
    name: nginx
    state: present

- name: Copy nginx config
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
  notify: Restart nginx

- name: Start nginx
  service:
    name: nginx
    state: started
    enabled: yes
```

```yaml
# roles/webserver/handlers/main.yml
---
- name: Restart nginx
  service:
    name: nginx
    state: restarted
```

```yaml
# roles/webserver/defaults/main.yml
---
nginx_port: 80
server_name: localhost
```

### Using Roles

```yaml
---
- name: Configure web servers
  hosts: webservers
  become: yes
  
  roles:
    - webserver
    - { role: database, db_name: myapp }
    - role: monitoring
      vars:
        alert_email: admin@example.com
```

## Ansible Vault

### Encrypting Files

```bash
# Create encrypted file
ansible-vault create secrets.yml

# Edit encrypted file
ansible-vault edit secrets.yml

# Encrypt existing file
ansible-vault encrypt vars.yml

# Decrypt file
ansible-vault decrypt vars.yml

# View encrypted file
ansible-vault view secrets.yml

# Change password
ansible-vault rekey secrets.yml
```

### Using Vault in Playbooks

```yaml
# secrets.yml (encrypted)
---
db_password: supersecret123
api_key: abc123xyz789
```

```bash
# Run playbook with vault
ansible-playbook playbook.yml --ask-vault-pass

# Use password file
ansible-playbook playbook.yml --vault-password-file ~/.vault_pass

# Multiple vault passwords
ansible-playbook playbook.yml --vault-id prod@~/.vault_prod
```

```yaml
# Using vaulted variables
---
- name: Deploy with secrets
  hosts: all
  vars_files:
    - secrets.yml
  
  tasks:
    - name: Use secret
      debug:
        msg: "DB password is {{ db_password }}"
      no_log: true
```

## Best Practices

### 1. Directory Structure

```
project/
├── ansible.cfg
├── inventory/
│   ├── production
│   ├── staging
│   ├── group_vars/
│   └── host_vars/
├── playbooks/
│   ├── site.yml
│   ├── webservers.yml
│   └── databases.yml
├── roles/
│   ├── common/
│   ├── webserver/
│   └── database/
├── templates/
├── files/
└── vars/
    └── secrets.yml
```

### 2. Use Roles

Organize playbooks into reusable roles.

### 3. Use Variables

Avoid hardcoding values.

### 4. Use Version Control

Store playbooks in Git.

### 5. Use ansible.cfg

```ini
# ansible.cfg
[defaults]
inventory = inventory/production
remote_user = ansible
private_key_file = ~/.ssh/ansible_key
host_key_checking = False
retry_files_enabled = False
gathering = smart
fact_caching = jsonfile
fact_caching_connection = /tmp/ansible_facts
fact_caching_timeout = 3600

[privilege_escalation]
become = True
become_method = sudo
become_user = root
become_ask_pass = False
```

### 6. Use Tags

```yaml
---
- name: System setup
  hosts: all
  
  tasks:
    - name: Install packages
      apt:
        name: "{{ item }}"
        state: present
      loop:
        - nginx
        - git
      tags:
        - packages
        - install
    
    - name: Configure firewall
      ufw:
        rule: allow
        port: 80
      tags:
        - security
        - firewall
```

```bash
# Run specific tags
ansible-playbook playbook.yml --tags "packages"

# Skip specific tags
ansible-playbook playbook.yml --skip-tags "firewall"
```

### 7. Use Check Mode

```bash
# Dry run
ansible-playbook playbook.yml --check

# Dry run with diff
ansible-playbook playbook.yml --check --diff
```

### 8. Use no_log for Sensitive Data

```yaml
- name: Set password
  user:
    name: myuser
    password: "{{ user_password }}"
  no_log: true
```

## Common Modules

```yaml
# File operations
- name: Create directory
  file:
    path: /opt/myapp
    state: directory
    mode: '0755'

# Copy file
- name: Copy file
  copy:
    src: /local/file
    dest: /remote/file
    owner: root
    mode: '0644'

# Template
- name: Deploy config
  template:
    src: config.j2
    dest: /etc/app/config.ini

# Package management
- name: Install package (apt)
  apt:
    name: nginx
    state: present
    update_cache: yes

# Service management
- name: Start service
  service:
    name: nginx
    state: started
    enabled: yes

# Command execution
- name: Run command
  command: /usr/bin/make_database.sh
  args:
    creates: /path/to/database

# Shell
- name: Run shell command
  shell: echo $HOME

# Git
- name: Clone repository
  git:
    repo: https://github.com/user/repo.git
    dest: /opt/app
    version: main

# User management
- name: Create user
  user:
    name: deploy
    uid: 1001
    groups: docker
    shell: /bin/bash

# Cron
- name: Add cron job
  cron:
    name: "backup database"
    minute: "0"
    hour: "2"
    job: "/usr/local/bin/backup.sh"
```

## Troubleshooting

### Debug Module

```yaml
- name: Debug variable
  debug:
    var: ansible_facts

- name: Debug message
  debug:
    msg: "The value is {{ my_var }}"

- name: Debug with verbosity
  debug:
    msg: "Detailed info"
    verbosity: 2
```

### Verbose Mode

```bash
# -v: verbose
# -vv: more verbose
# -vvv: debug
# -vvvv: connection debug
ansible-playbook playbook.yml -vvv
```

### Check Connectivity

```bash
ansible all -m ping
```

### Gather Facts

```bash
ansible hostname -m setup
```

## Useful Resources

- Official Documentation: https://docs.ansible.com/
- Galaxy (Roles): https://galaxy.ansible.com/
- GitHub: https://github.com/ansible/ansible
- Best Practices: https://docs.ansible.com/ansible/latest/tips_tricks/ansible_tips_tricks.html

## Quick Reference

| Command | Description |
|---------|-------------|
| `ansible all -m ping` | Test connectivity |
| `ansible-playbook playbook.yml` | Run playbook |
| `ansible-playbook --check` | Dry run |
| `ansible-playbook -vvv` | Debug mode |
| `ansible-vault create file` | Create encrypted file |
| `ansible-galaxy init role` | Create role |
| `ansible-inventory --list` | List inventory |
| `ansible all -m setup` | Gather facts |

---

*This guide covers Ansible fundamentals. For production use, consider implementing proper secret management, CI/CD integration, testing strategies (Molecule), and monitoring.*
