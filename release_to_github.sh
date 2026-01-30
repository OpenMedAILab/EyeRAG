#!/bin/bash

# Script to prepare EyeRAG for release on GitHub

echo "Preparing EyeRAG for GitHub release..."

# Verify the current status
echo "Current git status:"
git status --short

# Show the commit history
echo "Recent commits:"
git log --oneline -5

# Create a release tag
VERSION="v1.0.0"
echo "Creating release tag: $VERSION"
git tag -a "$VERSION" -m "Initial release of EyeRAG - Eye Retrieva-Augmented Generation system"

# Show the tag information
echo "Tag information:"
git show "$VERSION"

echo "EyeRAG is now ready for GitHub release!"
echo "To push to GitHub, run:"
echo "  git remote add origin <your_github_repo_url>"
echo "  git branch -M main"
echo "  git push -u origin main"
echo "  git push origin $VERSION"