#!/bin/bash

# mindthread-cli user installation script (no sudo required)

echo "🧠 Installing mindthread-cli to your home directory..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create ~/.local/bin if it doesn't exist
mkdir -p "$HOME/.local/bin"

# Create symlink
ln -sf "$SCRIPT_DIR/main.py" "$HOME/.local/bin/mindthread"
chmod +x "$HOME/.local/bin/mindthread"

echo "✅ Installed to $HOME/.local/bin/mindthread"

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo "⚠️  Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
    echo "Then run: source ~/.bashrc  (or restart your terminal)"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "You can now use 'mindthread' from anywhere:"
echo "  mindthread add \"your note here\""
echo "  mindthread list"
echo "  mindthread search \"query\""
echo "  mindthread show 1"
