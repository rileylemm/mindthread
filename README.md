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
| `add` | Add a new note (interactive mode with metadata + link review) |
| `add "note text"` | Add a new note (command line mode) |
| `list` | Show all saved notes |
| `search "query"` | Search through your notes |
| `show <id>` | Show a specific note in detail (pager + inline actions) |
| `related <id>` | Find related thoughts using AI embeddings |
| `stats` | Show note stats and sparkline history |
| `tags [limit]` | Display tag frequency heatmap |
| `ui` | Launch the optional prompt-toolkit TUI |
| `agent-brief` | Print an instant project orientation for AI agents |
| `catalog` | Review categories/tags and tidy the catalog |
| `clip` | Save current clipboard contents as a note |
| `help` | Show available commands |

Outputs for `list`, `search`, `stats`, and `tags` open in a pager by default. Add `--no-pager` to any of them to print directly.

`show <id>` also opens in a pager and accepts inline actions: `h` for help, `e` to edit the note text in your `$EDITOR`, `q` to close.

Launch the optional TUI with `mindthread ui`. It uses `prompt_toolkit`, keeps the tactile CLI vibe, and works on Python 3.13.

## Configuration

Set your OpenAI API key in the `.env` file:

```bash
OPENAI_API_KEY=sk-...
```

That's it! No other configuration needed.

## Architecture

- **Modular package**: CLI orchestrator with reusable modules in `mindthread_app/`
  - `cli.py` – command parsing, interactive prompts, console rendering
  - `notes.py` – note CRUD, search, related-note logic
  - `services/openai_service.py` – GPT metadata + embedding helpers with error handling
  - `storage.py` – JSON persistence (respects `DATA_DIR` and the legacy `notes.json` fallback)
  - `config.py` – dotenv-backed settings loader
  - `editor.py` – shared `$EDITOR` launcher for text edits
  - `analytics.py` – sparkline + heatmap render helpers
  - `promptui/` – prompt_toolkit-based optional interface (`run_ui`)
- **Storage**: JSON file (defaults to `notes.json`, or `<DATA_DIR>/notes.json` if configured)
- **Embeddings**: OpenAI text-embedding-3-small
- **AI Tagging**: GPT-4 for automatic categorization
- **Search**: Text matching + vector similarity with scikit-learn
- **Catalog**: `data/catalog.json` keeps known categories/tags for reuse and GPT prompting
- **CLI**: Lightweight command dispatcher (still installable via the `mindthread` script)
- **Clipboard helper**: `clip` command uses the system clipboard (`pbpaste` on macOS)
- **Analytics**: Sparkline stats (`stats`) and tag heatmap (`tags`) for quick at-a-glance context

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
- **Simple**: Small modular package, minimal dependencies, easy to understand
- **Auto-tagging**: No more manual categorization decisions
- **Personal**: Built for your own note-taking workflow
- **Extensible**: Easy to add features as you need them

## Future Enhancements

When you need more features, you can easily add:

- **FAISS Integration**: Scale to 1000+ notes with faster vector search
- **Edit**: Add note modification commands
- **Export**: Add JSON/Markdown export functions
- **Categories**: Add category filtering
- **Better CLI**: Add argparse for more options
- **Related Notes Limit**: Make the number of related notes configurable

The foundation is solid - vector search is already working with scikit-learn!

## Dependencies

Just 4 dependencies:
- `openai` - For embeddings and GPT tagging
- `python-dotenv` - For environment variable loading
- `scikit-learn` - For cosine similarity calculations
- `httpx` - Required by the OpenAI SDK

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
