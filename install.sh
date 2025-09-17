#!/bin/bash

# mindthread-cli installation script

echo "🧠 Installing mindthread-cli..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create a symlink in /usr/local/bin (requires sudo)
if [ -w "/usr/local/bin" ]; then
    ln -sf "$SCRIPT_DIR/mindthread" "/usr/local/bin/mindthread"
    chmod +x "/usr/local/bin/mindthread"
    echo "✅ Installed to /usr/local/bin/mindthread"
else
    echo "❌ Need sudo permissions to install to /usr/local/bin"
    echo "Run: sudo ./install.sh"
    exit 1
fi

echo "🎉 Installation complete!"
echo ""
echo "You can now use 'mindthread' from anywhere:"
echo "  mindthread add \"your note here\""
echo "  mindthread list"
echo "  mindthread search \"query\""
echo "  mindthread show 1"
