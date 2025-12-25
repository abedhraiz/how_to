# Regular Expressions (Regex) - Complete Guide

## What is Regex?

Regular Expressions (regex) are patterns used to match character combinations in strings. They're essential for:

- **Text searching and filtering**
- **Data validation** (emails, URLs, phone numbers)
- **Text processing** (find/replace, extraction)
- **Log analysis** and parsing
- **CI/CD pipeline filtering**
- **Git operations** (.gitignore, search)

---

## Table of Contents

- [Basic Syntax](#basic-syntax)
- [Character Classes](#character-classes)
- [Quantifiers](#quantifiers)
- [Anchors](#anchors)
- [Groups and Capturing](#groups-and-capturing)
- [Common Patterns](#common-patterns)
- [Regex in Different Tools](#regex-in-different-tools)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)
- [Testing and Debugging](#testing-and-debugging)

---

## Basic Syntax

### Literal Characters

```regex
hello       # Matches "hello"
123         # Matches "123"
hello world # Matches "hello world"
```

### Special Characters (Metacharacters)

These have special meaning and need escaping with `\` to match literally:

```
. ^ $ * + ? { } [ ] \ | ( )
```

**Examples:**
```regex
\.          # Matches literal dot "."
\$          # Matches literal dollar sign "$"
\*          # Matches literal asterisk "*"
```

### Wildcard

```regex
.           # Matches any single character (except newline)
h.t         # Matches "hat", "hit", "hot", "h9t", etc.
```

---

## Character Classes

### Predefined Classes

```regex
\d          # Any digit [0-9]
\D          # Any non-digit [^0-9]
\w          # Any word character [a-zA-Z0-9_]
\W          # Any non-word character [^a-zA-Z0-9_]
\s          # Any whitespace [ \t\n\r\f\v]
\S          # Any non-whitespace [^ \t\n\r\f\v]
```

**Examples:**
```regex
\d{3}       # Matches exactly 3 digits: "123"
\w+         # Matches one or more word characters: "hello"
\s*         # Matches zero or more whitespace
```

### Custom Classes

```regex
[abc]       # Matches 'a', 'b', or 'c'
[a-z]       # Matches any lowercase letter
[A-Z]       # Matches any uppercase letter
[0-9]       # Matches any digit
[a-zA-Z]    # Matches any letter
[^abc]      # Matches anything EXCEPT 'a', 'b', or 'c' (negation)
```

**Examples:**
```regex
[aeiou]     # Matches any vowel
[^aeiou]    # Matches any consonant
[0-9a-f]    # Matches hexadecimal digit
[A-Za-z0-9_-] # Matches alphanumeric plus underscore and hyphen
```

---

## Quantifiers

Specify how many times a pattern should match:

```regex
*           # 0 or more times
+           # 1 or more times
?           # 0 or 1 time (optional)
{n}         # Exactly n times
{n,}        # At least n times
{n,m}       # Between n and m times
```

**Examples:**
```regex
a*          # Matches "", "a", "aa", "aaa", etc.
a+          # Matches "a", "aa", "aaa", etc. (at least one)
a?          # Matches "" or "a"
a{3}        # Matches "aaa" (exactly 3)
a{2,4}      # Matches "aa", "aaa", or "aaaa"
\d{3,}      # Matches 3 or more digits
```

### Greedy vs Lazy (Non-Greedy)

```regex
.*          # Greedy: matches as much as possible
.*?         # Lazy: matches as little as possible
.+          # Greedy
.+?         # Lazy
```

**Example:**
```regex
Input: <div>Hello</div><div>World</div>

<.*>        # Matches: <div>Hello</div><div>World</div> (entire string)
<.*?>       # Matches: <div> (stops at first >)
```

---

## Anchors

Define position in the string:

```regex
^           # Start of string/line
$           # End of string/line
\b          # Word boundary
\B          # Non-word boundary
```

**Examples:**
```regex
^hello      # Matches "hello" only at start: "hello world" ✓
hello$      # Matches "hello" only at end: "say hello" ✓
^hello$     # Matches only if entire string is "hello"
\bhello\b   # Matches "hello" as a complete word (not "helloworld")
\Bhello\B   # Matches "hello" only within a word: "helloworld"
```

**Practical Examples:**
```bash
# Match lines starting with ERROR in logs
^ERROR.*

# Match email addresses at end of line
\S+@\S+\.\S+$

# Match standalone "cat" (not in "category")
\bcat\b
```

---

## Groups and Capturing

### Capturing Groups

```regex
(abc)       # Capture group - remembers matched text
(?:abc)     # Non-capturing group - groups without capturing
```

**Examples:**
```regex
(\d{3})-(\d{3})-(\d{4})    # Captures phone number parts
(https?):\/\/(.+)          # Captures protocol and domain
```

**Backreferences:**
```regex
\1, \2, \3  # Reference captured groups

(\w+)\s+\1  # Matches repeated words: "hello hello"
```

### Named Groups

```regex
(?<name>pattern)    # Named capturing group (Python, .NET, etc.)
```

**Example:**
```regex
(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})
# Matches: 2025-12-24
# Can reference as: \k<year>, \k<month>, \k<day>
```

---

## Common Patterns

### Email Validation

```regex
# Simple (basic validation)
[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}

# More comprehensive
^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$
```

**Examples:**
```
✓ user@example.com
✓ first.last@example.co.uk
✓ user+tag@example.com
✗ @example.com
✗ user@
✗ user@example
```

### URL Validation

```regex
# Simple URL
https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)

# With optional protocol
(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)
```

### Phone Numbers

```regex
# US Format: (123) 456-7890
\(\d{3}\)\s*\d{3}-\d{4}

# International: +1-123-456-7890
\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}

# Flexible US format
\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}
```

### IP Addresses

```regex
# IPv4 (simple)
\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}

# IPv4 (accurate - validates 0-255)
^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$
```

### Dates

```regex
# YYYY-MM-DD
\d{4}-\d{2}-\d{2}

# MM/DD/YYYY or M/D/YYYY
\d{1,2}\/\d{1,2}\/\d{4}

# DD-MM-YYYY
\d{2}-\d{2}-\d{4}
```

### Credit Cards

```regex
# Visa
^4[0-9]{12}(?:[0-9]{3})?$

# MasterCard
^5[1-5][0-9]{14}$

# American Express
^3[47][0-9]{13}$

# Generic (with spaces/dashes)
^[\d\s-]{13,19}$
```

### Passwords

```regex
# At least 8 chars, 1 uppercase, 1 lowercase, 1 digit
^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d]{8,}$

# At least 8 chars, 1 uppercase, 1 lowercase, 1 digit, 1 special char
^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$
```

### File Extensions

```regex
# Match .txt files
.*\.txt$

# Match image files
.*\.(jpg|jpeg|png|gif|webp)$

# Match without extension
^[^.]+$
```

### HTML Tags

```regex
# Match opening and closing tags
<([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)

# Simple tag matcher
<\/?[\w\s]*>|<.+[\W]>
```

---

## Regex in Different Tools

### 1. Grep (Linux/Unix)

```bash
# Basic grep
grep "pattern" file.txt

# Extended regex (-E)
grep -E "pattern|pattern2" file.txt

# Case insensitive (-i)
grep -i "error" log.txt

# Recursive (-r)
grep -r "TODO" ./src/

# Show line numbers (-n)
grep -n "function" script.py

# Invert match (-v) - show lines NOT matching
grep -v "^#" config.txt

# Count matches (-c)
grep -c "ERROR" log.txt
```

**Examples:**
```bash
# Find all Python files
find . -name "*.py" | grep -E "test_.*\.py$"

# Find ERROR or WARNING in logs
grep -E "ERROR|WARNING" app.log

# Find email addresses
grep -E "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" data.txt

# Find IP addresses
grep -E "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" access.log

# Find lines starting with digits
grep "^[0-9]" file.txt
```

### 2. Sed (Stream Editor)

```bash
# Basic substitution
sed 's/old/new/' file.txt

# Global substitution (all occurrences)
sed 's/old/new/g' file.txt

# In-place editing
sed -i 's/old/new/g' file.txt

# Delete lines matching pattern
sed '/pattern/d' file.txt

# Print only matching lines
sed -n '/pattern/p' file.txt
```

**Examples:**
```bash
# Replace all occurrences of foo with bar
sed 's/foo/bar/g' input.txt

# Remove trailing whitespace
sed 's/[[:space:]]*$//' file.txt

# Remove empty lines
sed '/^$/d' file.txt

# Comment out lines containing "debug"
sed 's/.*debug.*/# &/' config.txt

# Extract email from text
echo "Contact: user@example.com" | sed -E 's/.*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}).*/\1/'
```

### 3. Awk

```bash
# Print lines matching pattern
awk '/pattern/' file.txt

# Print specific field
awk '{print $1}' file.txt

# Pattern with action
awk '/ERROR/ {print $1, $2}' log.txt
```

**Examples:**
```bash
# Print 2nd column of CSV
awk -F',' '{print $2}' data.csv

# Sum numbers in first column
awk '{sum += $1} END {print sum}' numbers.txt

# Print lines where 3rd field is greater than 100
awk '$3 > 100' data.txt

# Complex pattern matching
awk '/^ERROR/ && /database/ {print $0}' log.txt
```

### 4. Python

```python
import re

# Basic matching
match = re.search(r'pattern', string)
if match:
    print(match.group())

# Find all matches
matches = re.findall(r'\d+', 'There are 123 apples and 456 oranges')
# Returns: ['123', '456']

# Replace
result = re.sub(r'old', 'new', string)

# Split
parts = re.split(r'\s+', 'split  by   whitespace')

# Compile for reuse
pattern = re.compile(r'\d+')
matches = pattern.findall(string)
```

**Examples:**
```python
# Email validation
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

# Extract phone numbers
text = "Call me at 555-123-4567 or 555-987-6543"
phones = re.findall(r'\d{3}-\d{3}-\d{4}', text)
# Returns: ['555-123-4567', '555-987-6543']

# Parse log lines
log_pattern = r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) (\w+): (.+)'
match = re.match(log_pattern, '2025-12-24 10:30:45 ERROR: Database connection failed')
if match:
    date, time, level, message = match.groups()

# Replace multiple spaces with single space
text = re.sub(r'\s+', ' ', 'too    many     spaces')

# Named groups
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
match = re.match(pattern, '2025-12-24')
print(match.group('year'))  # '2025'
```

### 5. JavaScript

```javascript
// Basic matching
const pattern = /pattern/;
const match = string.match(pattern);

// Global flag
const matches = string.match(/pattern/g);

// Replace
const result = string.replace(/old/g, 'new');

// Test
if (/\d+/.test(string)) {
    console.log('Contains digits');
}

// Split
const parts = string.split(/\s+/);
```

**Examples:**
```javascript
// Email validation
function validateEmail(email) {
    const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    return pattern.test(email);
}

// Extract URLs
const text = "Visit https://example.com or http://test.org";
const urls = text.match(/https?:\/\/[^\s]+/g);
// Returns: ['https://example.com', 'http://test.org']

// Replace with function
const result = "hello world".replace(/\w+/g, match => match.toUpperCase());
// Returns: "HELLO WORLD"

// Named capture groups (ES2018+)
const pattern = /(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})/;
const match = '2025-12-24'.match(pattern);
console.log(match.groups.year);  // '2025'
```

### 6. Git

```bash
# .gitignore patterns
*.log           # All .log files
node_modules/   # Directory
**/temp/        # temp directory at any level
*.txt           # All .txt files
!important.txt  # Exception (don't ignore this)

# Git grep
git grep "pattern"
git grep -E "pattern1|pattern2"
git grep -n "TODO"  # Show line numbers

# Git log
git log --grep="fix.*bug"  # Commits with "fix...bug" in message
git log --author="John.*Doe"  # Commits by author matching pattern
```

### 7. VS Code

```json
// Search with regex (Enable regex in search box)
"search.smartCase": true,
"search.useRegex": true

// Find: \bfunction\s+\w+\s*\(
// Replace: const $& = 

// Multi-line search
"search.multiline": true
```

---

## Real-World Examples

### 1. Log Analysis

```bash
# Find all errors in last hour
grep -E "$(date -d '1 hour ago' '+%Y-%m-%d %H').*ERROR" /var/log/app.log

# Extract IP addresses from access logs
grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" access.log | sort -u

# Find 404 errors with IP
grep "HTTP/1\.[01]\" 404" access.log | grep -oE "^[0-9.]+"

# Count errors by type
grep -oE "ERROR.*" app.log | sed 's/ERROR: //' | sort | uniq -c | sort -rn
```

### 2. Data Extraction

```python
# Extract data from HTML
import re

html = '<div class="price">$19.99</div>'
price = re.search(r'\$(\d+\.\d{2})', html).group(1)
# Returns: '19.99'

# Parse CSV with quotes
line = 'John,"Doe, Jr.",30,New York'
fields = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', line)
# Returns: ['John', '"Doe, Jr."', '30', 'New York']

# Extract hashtags
text = "Great day! #python #coding #regex"
hashtags = re.findall(r'#\w+', text)
# Returns: ['#python', '#coding', '#regex']
```

### 3. File Processing

```bash
# Rename files: convert spaces to underscores
for file in *\ *; do
    mv "$file" "${file// /_}"
done

# Find all TODO comments in code
grep -rn "TODO:" --include="*.py" --include="*.js" .

# Remove comments from config files
sed '/^[[:space:]]*#/d' config.ini

# Extract function names from Python files
grep -oE "def\s+\w+\s*\(" *.py | sed 's/def \(\w\+\).*/\1/'
```

### 4. GitHub Actions (From Our Workflows)

```yaml
# Path filtering in workflows
on:
  push:
    paths:
      - '**.md'           # Any markdown file
      - 'docs/**'         # Anything in docs/
      - '.github/workflows/*.yml'  # Workflow files

# Label matching in auto-label-pr.yml
const labelRules = {
  'documentation': /\.md$/,
  'github-actions': /\.github\/workflows\//,
};
```

### 5. Data Validation

```python
# Validate username (alphanumeric, 3-16 chars)
def validate_username(username):
    return bool(re.match(r'^[a-zA-Z0-9_]{3,16}$', username))

# Validate strong password
def validate_password(password):
    # At least 8 chars, 1 upper, 1 lower, 1 digit, 1 special
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return bool(re.match(pattern, password))

# Validate hexadecimal color
def validate_color(color):
    return bool(re.match(r'^#[0-9a-fA-F]{6}$', color))

# Validate credit card (Luhn algorithm separate)
def validate_credit_card_format(number):
    # Remove spaces/dashes
    number = re.sub(r'[\s-]', '', number)
    return bool(re.match(r'^\d{13,19}$', number))
```

### 6. Text Cleaning

```python
# Remove HTML tags
def strip_html(text):
    return re.sub(r'<[^>]+>', '', text)

# Remove extra whitespace
def clean_whitespace(text):
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to one
    return text.strip()

# Remove special characters (keep alphanumeric and spaces)
def remove_special_chars(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Normalize phone number format
def normalize_phone(phone):
    digits = re.sub(r'\D', '', phone)  # Remove non-digits
    if len(digits) == 10:
        return f'({digits[:3]}) {digits[3:6]}-{digits[6:]}'
    return phone
```

---

## Best Practices

### 1. Use Raw Strings (Python)

```python
# Good - use raw string
pattern = r'\d+\.\d+'

# Bad - requires double escaping
pattern = '\\d+\\.\\d+'
```

### 2. Compile Regex for Reuse

```python
# Efficient for multiple uses
email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

for email in email_list:
    if email_pattern.match(email):
        print(f"Valid: {email}")
```

### 3. Use Non-Capturing Groups When Possible

```python
# Non-capturing is more efficient
pattern = r'(?:https?|ftp)://\S+'  # Non-capturing

# vs capturing (when you don't need the capture)
pattern = r'(https?|ftp)://\S+'    # Capturing
```

### 4. Be Specific

```regex
# Good - specific
^\d{3}-\d{3}-\d{4}$

# Bad - too broad
.*\d.*-.*\d.*-.*\d.*
```

### 5. Comment Complex Patterns

```python
# Use verbose mode for complex patterns
pattern = re.compile(r'''
    ^                   # Start of string
    (?P<protocol>https?) # Protocol (http or https)
    ://                 # ://
    (?P<domain>[\w.-]+) # Domain name
    (?P<path>/[\w/.-]*) # Optional path
    $                   # End of string
''', re.VERBOSE)
```

### 6. Escape User Input

```python
import re

user_input = "user@example.com"
safe_pattern = re.escape(user_input)
# Escapes special characters to treat as literal
```

### 7. Test Edge Cases

```python
# Test various inputs
test_cases = [
    "valid@example.com",       # Valid
    "@example.com",            # Invalid - no username
    "user@",                   # Invalid - no domain
    "user@example",            # Invalid - no TLD
    "user@example.co.uk",      # Valid - multiple TLDs
    "user+tag@example.com",    # Valid - plus addressing
]
```

### 8. Avoid Catastrophic Backtracking

```regex
# Bad - can cause exponential time
^(a+)+b

# Good - more efficient
^a+b

# Bad - nested quantifiers
(.*)*

# Good - specific pattern
[\w\s]*
```

---

## Testing and Debugging

### Online Tools

1. **[regex101.com](https://regex101.com/)**
   - Real-time testing
   - Explanation of pattern
   - Quick reference
   - Multiple flavors (Python, JavaScript, etc.)

2. **[regexr.com](https://regexr.com/)**
   - Visual pattern builder
   - Community patterns
   - Cheat sheet

3. **[regexpal.com](https://www.regexpal.com/)**
   - Simple and fast
   - Good for quick tests

### Command-Line Testing

```bash
# Test with grep
echo "test string" | grep -E "pattern"

# Test with Python one-liner
python -c "import re; print(re.findall(r'pattern', 'test string'))"

# Test with sed
echo "test string" | sed -E 's/pattern/replacement/'
```

### Python Testing Script

```python
import re

def test_regex(pattern, test_strings):
    """Test regex pattern against multiple strings"""
    compiled = re.compile(pattern)
    
    print(f"Pattern: {pattern}\n")
    for test_str in test_strings:
        match = compiled.search(test_str)
        status = "✓" if match else "✗"
        result = match.group() if match else "No match"
        print(f"{status} '{test_str}' -> {result}")

# Example usage
pattern = r'\b\d{3}-\d{3}-\d{4}\b'
tests = [
    "Call me at 555-123-4567",
    "My number is 123-456-7890",
    "Invalid: 12-345-6789",
    "No number here",
]

test_regex(pattern, tests)
```

### Debugging Tips

1. **Start Simple**
   ```regex
   # Build incrementally
   \d           # Match digit
   \d+          # Match multiple digits
   \d{3}        # Exactly 3 digits
   \d{3}-\d{3}  # Phone number pattern
   ```

2. **Test Components Separately**
   ```python
   # Break complex pattern into parts
   date_part = r'\d{4}-\d{2}-\d{2}'
   time_part = r'\d{2}:\d{2}:\d{2}'
   full_pattern = f'{date_part} {time_part}'
   ```

3. **Use Print Debugging**
   ```python
   pattern = r'(\w+)@(\w+)\.(\w+)'
   match = re.search(pattern, 'user@example.com')
   if match:
       print(f"Groups: {match.groups()}")
       print(f"Full match: {match.group(0)}")
   ```

4. **Check for Greedy Behavior**
   ```python
   # Greedy vs lazy
   text = '<div>Hello</div><div>World</div>'
   
   greedy = re.search(r'<.*>', text)
   print(f"Greedy: {greedy.group()}")
   # Output: <div>Hello</div><div>World</div>
   
   lazy = re.search(r'<.*?>', text)
   print(f"Lazy: {lazy.group()}")
   # Output: <div>
   ```

---

## Performance Considerations

### 1. Avoid Catastrophic Backtracking

```regex
# Bad - exponential time complexity
(a+)+b

# Test case that hangs:
"aaaaaaaaaaaaaaaaaaaaaaaaaaaa"

# Good - linear time
a+b
```

### 2. Use Anchors

```regex
# Slower - checks entire string
\d{3}-\d{3}-\d{4}

# Faster - anchors reduce search space
^\d{3}-\d{3}-\d{4}$
```

### 3. Use Character Classes

```regex
# Faster
[0-9]

# Slower (in some engines)
(0|1|2|3|4|5|6|7|8|9)
```

### 4. Compile Once, Use Many Times

```python
# Slow - compiles every iteration
for text in texts:
    if re.match(r'pattern', text):
        process(text)

# Fast - compile once
pattern = re.compile(r'pattern')
for text in texts:
    if pattern.match(text):
        process(text)
```

---

## Quick Reference Cheat Sheet

### Basics
```
.       Any character except newline
^       Start of string/line
$       End of string/line
\       Escape special character
|       Alternation (OR)
```

### Character Classes
```
[abc]   a, b, or c
[^abc]  Not a, b, or c
[a-z]   Range (lowercase letters)
\d      Digit [0-9]
\D      Non-digit [^0-9]
\w      Word char [a-zA-Z0-9_]
\W      Non-word char
\s      Whitespace [ \t\n\r\f\v]
\S      Non-whitespace
```

### Quantifiers
```
*       0 or more
+       1 or more
?       0 or 1
{n}     Exactly n
{n,}    n or more
{n,m}   Between n and m
*?      Lazy 0 or more
+?      Lazy 1 or more
```

### Groups
```
(...)   Capturing group
(?:...) Non-capturing group
\1      Backreference to group 1
(?P<name>...) Named group (Python)
```

### Anchors & Boundaries
```
^       Start of string/line
$       End of string/line
\b      Word boundary
\B      Non-word boundary
```

### Lookahead/Lookbehind
```
(?=...)   Positive lookahead
(?!...)   Negative lookahead
(?<=...)  Positive lookbehind
(?<!...)  Negative lookbehind
```

---

## Related Documentation

- [Git Guide](../git/git-guide.md) - Use regex in .gitignore and git grep
- [GitHub Actions Guide](../../cicd-automation/github-actions/github-actions-guide.md) - Path filtering with regex
- [Apache Airflow Guide](../../data-engineering/workflow-orchestration/apache-airflow-guide.md) - Log parsing
- [Apache Kafka Guide](../../data-engineering/streaming/apache-kafka-guide.md) - Message filtering

---

## Additional Resources

### Documentation
- [Python re module](https://docs.python.org/3/library/re.html)
- [JavaScript RegExp](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions)
- [grep manual](https://www.gnu.org/software/grep/manual/grep.html)

### Interactive Learning
- [RegexOne](https://regexone.com/) - Interactive tutorial
- [Regex Crossword](https://regexcrossword.com/) - Practice with puzzles
- [Regular-Expressions.info](https://www.regular-expressions.info/) - Comprehensive guide

### Testing Tools
- [regex101](https://regex101.com/) - Online tester with explanation
- [RegExr](https://regexr.com/) - Visual regex builder
- [RegexPal](https://www.regexpal.com/) - Quick testing

### Books
- "Mastering Regular Expressions" by Jeffrey Friedl
- "Regular Expressions Cookbook" by Jan Goyvaerts

---

**Last Updated**: December 2025  
**Maintainers**: Documentation Team  
**Version**: 1.0
