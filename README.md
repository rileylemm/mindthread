# mindthread-cli

**A lightweight terminal-first assistant for building your second brain.**

Add thoughts. Embed them. Let GPT tag and organize. Explore the connections.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   cp env.example .env
   # Edit .env and add your OpenAI API key
   ```

3. **Install the mindthread command:**
   ```bash
   sudo ./install.sh
   # or for user installation: ./install-user.sh
   ```

4. **Add your first note:**
   ```bash
   mindthread add
   # Follow the interactive prompts
   ```

5. **List your notes:**
   ```bash
   mindthread list
   ```

## Commands

| Command | Description |
|---------|-------------|
| `add` | Add a new note (interactive mode) |
| `add "note text"` | Add a new note (command line mode) |
| `list` | Show all saved notes |
| `search "query"` | Search through your notes |
| `show <id>` | Show a specific note in detail |
| `related <id>` | Find related thoughts using AI embeddings |
| `help` | Show available commands |

## Configuration

Set your OpenAI API key in the `.env` file:

```bash
OPENAI_API_KEY=sk-...
```

That's it! No other configuration needed.

## Architecture

- **Single file**: Everything in `main.py` (~200 lines)
- **Storage**: Simple JSON file (`notes.json`)
- **Embeddings**: OpenAI text-embedding-3-small
- **AI Tagging**: GPT-4 for automatic categorization
- **Search**: Text matching + vector similarity with scikit-learn
- **CLI**: Simple `sys.argv` parsing

## Example Workflow

```bash
$ mindthread add
Add your note here: I want to find a way to let my notes connect themselves into a web of ideas

Processing note...

Generated metadata:
Title: Web of Ideas
Category: Knowledge Systems
Tags: notes, connections, creativity, networks

Confirm category/tags? (y/n/edit): y

Generating embedding...
Note saved! ID: 1

$ mindthread list

Your Notes (1 total):
==================================================

[1] Web of Ideas
Category: Knowledge Systems
Tags: notes, connections, creativity, networks
Created: 2025-01-27
Text: I want to find a way to let my notes connect themselves into a web of ideas.
------------------------------

$ mindthread search "connect ideas"
Found 1 matching notes:
==================================================

[1] Web of Ideas
Category: Knowledge Systems
Text: I want to find a way to let my notes connect themselves into a web of ideas.
------------------------------

$ mindthread related 1
🧠 Related thoughts for: Web of Ideas
============================================================
Target note: I want to find a way to let my notes connect themselves into a web of ideas

Most similar notes:
----------------------------------------

1. [2] Building Knowledge Networks (similarity: 0.847)
   Category: Knowledge Systems
   Tags: networks, learning, connections, systems
   Text: Creating interconnected knowledge systems that help ideas flow...
----------------------------------------
```

## Why This Project?

- **Low friction**: Just type and let AI handle the organization
- **Terminal-first**: No GUI distractions, pure thought capture
- **Simple**: Single file, minimal dependencies, easy to understand
- **Auto-tagging**: No more manual categorization decisions
- **Personal**: Built for your own note-taking workflow
- **Extensible**: Easy to add features as you need them

## Future Enhancements

When you need more features, you can easily add:

- **FAISS Integration**: Scale to 1000+ notes with faster vector search
- **Edit/Delete**: Add note modification commands
- **Export**: Add JSON/Markdown export functions
- **Categories**: Add category filtering
- **Better CLI**: Add argparse for more options
- **Related Notes Limit**: Make the number of related notes configurable

The foundation is solid - vector search is already working with scikit-learn!

## Dependencies

Just 3 dependencies:
- `openai` - For embeddings and GPT tagging
- `python-dotenv` - For environment variable loading
- `scikit-learn` - For cosine similarity calculations

## Installation

### **Option 1: Global Installation (Recommended)**

```bash
git clone <your-repo>
cd mindthread
pip install -r requirements.txt
cp env.example .env
# Add your OpenAI API key to .env
sudo ./install.sh
```

### **Option 2: User Installation (No sudo required)**


git clone <your-repo>
cd mindthread
pip install -r requirements.txt
cp env.example .env
# Add your OpenAI API key to .env
./install-user.sh
# Follow the PATH instructions if needed
```

### **Option 3: Manual Installation**

```bash
git clone <your-repo>
cd mindthread
pip install -r requirements.txt
cp env.example .env
# Add your OpenAI API key to .env
# Create symlink manually:
ln -sf $(pwd)/mindthread /usr/local/bin/mindthread
chmod +x /usr/local/bin/mindthread
```

After installation, you can use `mindthread` from anywhere in your terminal!

```bash
# Interactive mode (recommended)
mindthread add

# Command line mode
mindthread add "your note here"
```

That's it! You're ready to start building your second brain.