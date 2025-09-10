#!/usr/bin/env python3
"""
mindthread-cli: A lightweight terminal-first assistant for building your second brain.
Single file, minimal dependencies, maximum simplicity.
"""

import sys
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment
load_dotenv()

# Configuration
NOTES_FILE = "notes.json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4"

def load_notes():
    """Load notes from JSON file"""
    if not Path(NOTES_FILE).exists():
        return []
    with open(NOTES_FILE, 'r') as f:
        return json.load(f)

def save_notes(notes):
    """Save notes to JSON file"""
    with open(NOTES_FILE, 'w') as f:
        json.dump(notes, f, indent=2)

def generate_embedding(text):
    """Generate OpenAI embedding for text"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def generate_metadata(text):
    """Generate title, category, and tags using GPT"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    prompt = f"""
Given this note, return JSON with:
- title: Short descriptive title (max 5 words)
- category: Single category name
- tags: Array of 3-5 relevant tags

Note: "{text}"

Return only JSON:
"""
    
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    
    try:
        return json.loads(response.choices[0].message.content.strip())
    except:
        return {"title": "Untitled", "category": "General", "tags": ["untagged"]}

def add_note_interactive():
    """Interactive note addition with AI processing and confirmation"""
    # Get note text from user
    text = input("Add your note here: ").strip()
    if not text:
        print("❌ Note text cannot be empty")
        return
    
    print("\nProcessing note...")
    
    # Generate metadata
    metadata = generate_metadata(text)
    
    # Display results
    print(f"\n📝 Generated metadata:")
    print(f"Title: {metadata['title']}")
    print(f"Category: {metadata['category']}")
    print(f"Tags: {', '.join(metadata['tags'])}")
    
    # Confirmation loop
    while True:
        choice = input("\nConfirm category/tags? (y/n/edit): ").strip().lower()
        
        if choice == 'y':
            # User confirmed, proceed with saving
            break
        elif choice == 'n':
            print("❌ Note not saved")
            return
        elif choice == 'edit':
            # Allow editing
            print(f"\nCurrent category: {metadata['category']}")
            new_category = input("New category (or press enter to keep current): ").strip()
            if new_category:
                metadata['category'] = new_category
            
            print(f"Current tags: {', '.join(metadata['tags'])}")
            new_tags = input("New tags (comma-separated, or press enter to keep current): ").strip()
            if new_tags:
                metadata['tags'] = [tag.strip() for tag in new_tags.split(',')]
            
            print(f"\nUpdated metadata:")
            print(f"Title: {metadata['title']}")
            print(f"Category: {metadata['category']}")
            print(f"Tags: {', '.join(metadata['tags'])}")
        else:
            print("Please enter 'y' (yes), 'n' (no), or 'edit'")
    
    # Generate embedding
    print("\nGenerating embedding...")
    embedding = generate_embedding(text)
    
    # Create note
    note = {
        "id": str(len(load_notes()) + 1),
        "text": text,
        "title": metadata["title"],
        "category": metadata["category"],
        "tags": metadata["tags"],
        "embedding": embedding,
        "created_at": datetime.now().isoformat()
    }
    
    # Save
    notes = load_notes()
    notes.append(note)
    save_notes(notes)
    
    print(f"\n✅ Note saved! ID: {note['id']}")

def add_note(text):
    """Add a new note with AI processing (legacy function for command line args)"""
    print("Processing note...")
    
    # Generate metadata
    metadata = generate_metadata(text)
    print(f"Title: {metadata['title']}")
    print(f"Category: {metadata['category']}")
    print(f"Tags: {', '.join(metadata['tags'])}")
    
    # Generate embedding
    embedding = generate_embedding(text)
    
    # Create note
    note = {
        "id": str(len(load_notes()) + 1),
        "text": text,
        "title": metadata["title"],
        "category": metadata["category"],
        "tags": metadata["tags"],
        "embedding": embedding,
        "created_at": datetime.now().isoformat()
    }
    
    # Save
    notes = load_notes()
    notes.append(note)
    save_notes(notes)
    
    print(f"✅ Note saved! ID: {note['id']}")

def list_notes():
    """List all notes"""
    notes = load_notes()
    if not notes:
        print("No notes found.")
        return
    
    print(f"\n📝 Your Notes ({len(notes)} total):")
    print("=" * 50)
    
    for note in notes:
        print(f"\n[{note['id']}] {note['title']}")
        print(f"Category: {note['category']}")
        print(f"Tags: {', '.join(note['tags'])}")
        print(f"Created: {note['created_at'][:10]}")
        print(f"Text: {note['text'][:100]}{'...' if len(note['text']) > 100 else ''}")
        print("-" * 30)

def search_notes(query):
    """Simple text search through notes"""
    notes = load_notes()
    if not notes:
        print("No notes found.")
        return
    
    # Simple text matching
    matches = []
    query_lower = query.lower()
    
    for note in notes:
        if (query_lower in note['text'].lower() or 
            query_lower in note['title'].lower() or
            any(query_lower in tag.lower() for tag in note['tags'])):
            matches.append(note)
    
    if not matches:
        print("No matching notes found.")
        return
    
    print(f"\n🔍 Found {len(matches)} matching notes:")
    print("=" * 50)
    
    for note in matches:
        print(f"\n[{note['id']}] {note['title']}")
        print(f"Category: {note['category']}")
        print(f"Text: {note['text']}")
        print("-" * 30)

def show_note(note_id):
    """Show a specific note"""
    notes = load_notes()
    note = next((n for n in notes if n['id'] == note_id), None)
    
    if not note:
        print(f"Note {note_id} not found.")
        return
    
    print(f"\n📝 {note['title']}")
    print("=" * 50)
    print(f"Category: {note['category']}")
    print(f"Tags: {', '.join(note['tags'])}")
    print(f"Created: {note['created_at']}")
    print(f"\nText:\n{note['text']}")

def find_related_notes(note_id, top_k=5):
    """Find related notes using cosine similarity on embeddings"""
    notes = load_notes()
    if not notes:
        print("No notes found.")
        return
    
    # Find the target note
    target_note = next((n for n in notes if n['id'] == note_id), None)
    if not target_note:
        print(f"Note {note_id} not found.")
        return
    
    # Check if target note has embedding
    if 'embedding' not in target_note or not target_note['embedding']:
        print(f"Note {note_id} doesn't have an embedding. Cannot find related notes.")
        return
    
    # Get all notes with embeddings (excluding the target note)
    notes_with_embeddings = [n for n in notes if n['id'] != note_id and 'embedding' in n and n['embedding']]
    
    if not notes_with_embeddings:
        print("No other notes with embeddings found.")
        return
    
    # Convert embeddings to numpy arrays
    target_embedding = np.array(target_note['embedding']).reshape(1, -1)
    other_embeddings = np.array([n['embedding'] for n in notes_with_embeddings])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(target_embedding, other_embeddings)[0]
    
    # Get top k most similar notes
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\n🧠 Related thoughts for: {target_note['title']}")
    print("=" * 60)
    print(f"Target note: {target_note['text'][:100]}{'...' if len(target_note['text']) > 100 else ''}")
    print("\nMost similar notes:")
    print("-" * 40)
    
    for i, idx in enumerate(top_indices, 1):
        note = notes_with_embeddings[idx]
        similarity = similarities[idx]
        print(f"\n{i}. [{note['id']}] {note['title']} (similarity: {similarity:.3f})")
        print(f"   Category: {note['category']}")
        print(f"   Tags: {', '.join(note['tags'])}")
        print(f"   Text: {note['text'][:150]}{'...' if len(note['text']) > 150 else ''}")
        print("-" * 40)

def show_help():
    """Show available commands"""
    print("🧠 mindthread-cli - Build your second brain")
    print("\nCommands:")
    print("  add                 - Add a new note (interactive)")
    print("  add \"note text\"     - Add a new note (command line)")
    print("  list                - List all notes")
    print("  search \"query\"      - Search notes")
    print("  show <id>           - Show specific note")
    print("  related <id>        - Find related thoughts using AI embeddings")
    print("  help                - Show this help message")

def main():
    """Main CLI entry point"""
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("Create a .env file with: OPENAI_API_KEY=sk-...")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        show_help()
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "add":
        if len(sys.argv) < 3:
            # Interactive mode - no arguments provided
            add_note_interactive()
        else:
            # Command line mode - text provided as argument
            text = sys.argv[2]
            add_note(text)
    
    elif command == "list":
        list_notes()
    
    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: python main.py search \"your query\"")
            sys.exit(1)
        query = sys.argv[2]
        search_notes(query)
    
    elif command == "show":
        if len(sys.argv) < 3:
            print("Usage: python main.py show <note_id>")
            sys.exit(1)
        note_id = sys.argv[2]
        show_note(note_id)
    
    elif command == "related":
        if len(sys.argv) < 3:
            print("Usage: python main.py related <note_id>")
            sys.exit(1)
        note_id = sys.argv[2]
        find_related_notes(note_id)
    
    elif command == "help":
        show_help()
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'help' to see available commands")
        sys.exit(1)

if __name__ == "__main__":
    main()
