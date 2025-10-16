#!/usr/bin/env python3
"""
Google Colab GitHub Authentication Setup

This script securely stores your GitHub token in Colab's environment
for authenticated git operations (pushing trained models).

Usage in Google Colab:
    !python setup_colab_auth.py
"""

import os
from getpass import getpass

def setup_github_auth():
    """Setup GitHub authentication for Colab"""
    
    print("=" * 60)
    print("GitHub Authentication Setup for Google Colab")
    print("=" * 60)
    print()
    
    # Get GitHub token securely (won't show in output)
    print("Enter your GitHub Personal Access Token:")
    print("(Get it from: https://github.com/settings/tokens)")
    print()
    token = getpass("Token: ")
    
    if not token:
        print("❌ No token provided. Aborting.")
        return False
    
    # Set environment variable
    os.environ['GITHUB_TOKEN'] = token
    
    # Also set for git credential helper
    username = input("\nEnter your GitHub username: ").strip()
    
    if not username:
        print("❌ No username provided. Aborting.")
        return False
    
    # Configure git to use token
    os.system('git config --global credential.helper store')
    
    # Create git credentials file
    repo_url = "https://github.com/giacatnk/keybot.git"
    credentials = f"https://{username}:{token}@github.com"
    
    # Store credentials
    cred_file = os.path.expanduser('~/.git-credentials')
    with open(cred_file, 'w') as f:
        f.write(f"{credentials}\n")
    
    print()
    print("✅ GitHub authentication configured!")
    print(f"   Username: {username}")
    print(f"   Repository: {repo_url}")
    print()
    print("You can now push to GitHub without entering credentials.")
    print()
    
    # Export to shell environment (for current session)
    print("Run this in your notebook to set environment variables:")
    print(f'    import os')
    print(f'    os.environ["GITHUB_TOKEN"] = "{token[:4]}...{token[-4:]}"')
    print()
    
    return True

if __name__ == "__main__":
    setup_github_auth()

