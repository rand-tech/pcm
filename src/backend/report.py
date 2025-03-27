import datetime
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, abort, jsonify, render_template, request

app = Flask(__name__)


def get_db_path():
    user_dir = Path.home()
    if os.name == 'nt':  # Windows
        db_path = user_dir / "AppData" / "Local" / "IDA_MCP"
    else:  # Linux/Mac
        db_path = user_dir / ".ida_mcp"

    db_path.mkdir(exist_ok=True)
    db_file = db_path / "analysis_notes.db"
    return str(db_file)


NOTES_DB = get_db_path()


def dict_factory(cursor, row):
    """Convert sqlite row to dictionary"""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def get_connection():
    """Get database connection with row factory set to return dictionaries"""
    conn = sqlite3.connect(NOTES_DB)
    conn.row_factory = dict_factory
    return conn


@app.route('/')
def index():
    """Main page showing list of analyzed files"""
    return render_template('index.html')


@app.route('/api/files')
def list_files():
    """API endpoint to get list of all analyzed files"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT f.*,
                   COUNT(n.id) as note_count,
                   MAX(n.timestamp) as last_note_timestamp
            FROM files f
            LEFT JOIN notes n ON f.md5 = n.file_md5
            GROUP BY f.md5
            ORDER BY f.last_accessed DESC
        """
        )

        files = cursor.fetchall()

        for file in files:
            if file['last_accessed']:
                file['last_accessed_formatted'] = datetime.datetime.fromtimestamp(file['last_accessed']).strftime('%Y-%m-%d %H:%M:%S')
            if file['last_note_timestamp']:
                file['last_note_formatted'] = datetime.datetime.fromtimestamp(file['last_note_timestamp']).strftime('%Y-%m-%d %H:%M:%S')

        conn.close()
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/files/<md5>/notes')
def get_file_notes(md5):
    """API endpoint to get notes for a specific file"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM files WHERE md5 = ?", (md5,))
        file_info = cursor.fetchone()

        if not file_info:
            conn.close()
            return jsonify({"error": "File not found"}), 404

        # Get notes
        cursor.execute(
            """
            SELECT * FROM notes
            WHERE file_md5 = ?
            ORDER BY timestamp DESC
        """,
            (md5,),
        )

        notes = cursor.fetchall()

        # Format timestamps and parse tags
        for note in notes:
            note['timestamp_formatted'] = datetime.datetime.fromtimestamp(note['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            if note['tags']:
                note['tags_list'] = [tag.strip() for tag in note['tags'].split(',')]
            else:
                note['tags_list'] = []

        conn.close()
        return jsonify({"file": file_info, "notes": notes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/notes/<int:note_id>')
def get_note_detail(note_id):
    """API endpoint to get details for a specific note"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT n.*, f.name as file_name, f.path as file_path
            FROM notes n
            JOIN files f ON n.file_md5 = f.md5
            WHERE n.id = ?
        """,
            (note_id,),
        )

        note = cursor.fetchone()

        if not note:
            conn.close()
            return jsonify({"error": "Note not found"}), 404

        note['timestamp_formatted'] = datetime.datetime.fromtimestamp(note['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        if note['tags']:
            note['tags_list'] = [tag.strip() for tag in note['tags'].split(',')]
        else:
            note['tags_list'] = []

        conn.close()
        return jsonify(note)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/tags')
def get_all_tags():
    """API endpoint to get all unique tags used across notes"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT tags FROM notes WHERE tags IS NOT NULL AND tags != ''")
        tag_rows = cursor.fetchall()

        # Process tags
        all_tags = set()
        for row in tag_rows:
            tags = row['tags'].split(',')
            for tag in tags:
                tag = tag.strip()
                if tag:
                    all_tags.add(tag)

        conn.close()
        return jsonify(sorted(list(all_tags)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/search')
def search_notes():
    """API endpoint to search notes by query term and/or tag"""
    try:
        query = request.args.get('q', '')
        tag = request.args.get('tag', '')

        conn = get_connection()
        cursor = conn.cursor()

        params = []
        sql = """
            SELECT n.*, f.name as file_name
            FROM notes n
            JOIN files f ON n.file_md5 = f.md5
            WHERE 1=1
        """

        if query:
            sql += " AND (n.title LIKE ? OR n.content LIKE ?)"
            params.extend([f'%{query}%', f'%{query}%'])

        if tag:
            sql += " AND n.tags LIKE ?"
            params.append(f'%{tag}%')

        sql += " ORDER BY n.timestamp DESC"

        cursor.execute(sql, params)
        notes = cursor.fetchall()

        # Format timestamps and parse tags
        for note in notes:
            note['timestamp_formatted'] = datetime.datetime.fromtimestamp(note['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            if note['tags']:
                note['tags_list'] = [tag.strip() for tag in note['tags'].split(',')]
            else:
                note['tags_list'] = []

        conn.close()
        return jsonify(notes)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print(f"Database path: {NOTES_DB}")

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name='notes' OR name='files')")
        tables = cursor.fetchall()
        table_names = [t['name'] for t in tables]

        if 'notes' not in table_names or 'files' not in table_names:
            print("Warning: Database exists but required tables are missing.")
            print("Make sure the IDA plugin has been run at least once to initialize the database.")
        else:
            # Get some basic stats
            cursor.execute("SELECT COUNT(*) as count FROM files")
            file_count = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM notes")
            note_count = cursor.fetchone()['count']

            print(f"Database contains {file_count} files and {note_count} notes.")

        conn.close()
    except Exception as e:
        print(f"Warning: Could not verify database: {str(e)}")
        print("Database will be created if it doesn't exist when you add your first note.")

    # Run the Flask app with debug mode enabled
    print("Starting web server...")
    PORT = 8000
    print(f"Open http://localhost:{PORT} in your web browser to view the reports")
    app.run(debug=True, host='localhost', port=PORT)
