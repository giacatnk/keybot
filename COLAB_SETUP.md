# Google Colab Setup Guide

## Quick Setup for Authenticated Git Push

### Step 1: Set GitHub Token in Colab

Run this in a Colab code cell (replace with your token):

```python
import os

# Set your GitHub token
GITHUB_TOKEN = "ghp_YOUR_TOKEN_HERE"  # Replace with your actual token
os.environ['GITHUB_TOKEN'] = GITHUB_TOKEN

# Configure git with token
GITHUB_USERNAME = "giacatnk"  # Your GitHub username
GITHUB_REPO = "keybot"

# Configure git credentials
!git config --global user.email "giacatnk@gmail.com"
!git config --global user.name "giacatnk"

# Set up credential helper to use token
!git config --global credential.helper store
!echo "https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com" > ~/.git-credentials

print("✅ Git authentication configured!")
print(f"   Username: {GITHUB_USERNAME}")
print(f"   Repository: https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git")
```

### Step 2: Clone Repository

```bash
!git clone https://github.com/giacatnk/keybot.git
%cd keybot
```

### Step 3: Run Training and Auto-Push

```bash
# Train and automatically push models to GitHub
!./setup_and_run.sh all -y
```

---

## Alternative: Use Colab Secrets (More Secure)

Google Colab has a built-in secrets manager:

1. Click the 🔑 key icon in the left sidebar
2. Add a new secret:
   - **Name:** `GITHUB_TOKEN`
   - **Value:** `ghp_YOUR_TOKEN_HERE`
3. Enable "Notebook access"

Then in your code:

```python
from google.colab import userdata
import os

# Get token from Colab secrets
GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
os.environ['GITHUB_TOKEN'] = GITHUB_TOKEN

# Configure git
GITHUB_USERNAME = "giacatnk"
!git config --global user.email "giacatnk@gmail.com"
!git config --global user.name "{GITHUB_USERNAME}"
!git config --global credential.helper store
!echo "https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com" > ~/.git-credentials

print("✅ Authenticated with GitHub!")
```

---

## Complete Colab Notebook Template

```python
# ===================================
# Cell 1: Setup Authentication
# ===================================
import os
from google.colab import userdata

# Get token (from secrets or set directly)
GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')  # Or use: "ghp_YOUR_TOKEN"
GITHUB_USERNAME = "giacatnk"

os.environ['GITHUB_TOKEN'] = GITHUB_TOKEN

# Configure git
!git config --global user.email "giacatnk@gmail.com"
!git config --global user.name "{GITHUB_USERNAME}"
!git config --global credential.helper store
!echo "https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com" > ~/.git-credentials

print("✅ Git authentication configured!")

# ===================================
# Cell 2: Clone Repository
# ===================================
!git clone https://github.com/giacatnk/keybot.git
%cd keybot

# ===================================
# Cell 3: Run Full Workflow
# ===================================
# This will: setup → train → evaluate → commit → push
!./setup_and_run.sh all -y

# ===================================
# Cell 4: Check Results (Optional)
# ===================================
!./setup_and_run.sh results
```

---

## Verify Authentication

Test if git push works:

```bash
# Test authentication
!git remote -v
!git config --list | grep user

# Try a test commit
!echo "test" > test.txt
!git add test.txt
!git commit -m "Test commit"
!git push origin main
```

If you see no password prompt and push succeeds, authentication is working! ✅

---

## Security Notes

⚠️ **Important:**
- **Never commit notebooks with tokens** to GitHub
- Use Colab Secrets instead of hardcoding tokens
- Revoke and regenerate tokens if accidentally exposed
- Keep your token private

---

## Troubleshooting

### Error: "Authentication failed"
```python
# Re-run authentication setup
!git config --global credential.helper store
!echo "https://giacatnk:YOUR_TOKEN@github.com" > ~/.git-credentials
```

### Error: "Permission denied"
- Check if token has `repo` scope enabled
- Generate new token at: https://github.com/settings/tokens

### Check if token is set:
```python
import os
print(os.environ.get('GITHUB_TOKEN', 'Not set'))
```

