import os
import json
import threading
from flask import Flask, request, jsonify
import psycopg
from openai import OpenAI
from dotenv import load_dotenv
import time
import uuid
import base64
import requests
from werkzeug.utils import secure_filename

# Load env variables (Ensure DATABASE_URL and OPENAI_API_KEY are in .env)
load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database connection string
DB_DSN = "postgresql://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/knowledge_DB"

# --- Helper: Embedding Generation ---
def generate_embedding(text):
    """Generates vector using OpenAI."""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ OpenAI Error: {e}")
        return None

# --- Background Task: Update Embeddings ---
def run_embedding_job(node_ids):
    """
    Runs in a separate thread. 
    Fetches the new nodes, generates embeddings, and updates the DB.
    """
    print(f"🔄 Starting background embedding job for {len(node_ids)} nodes...")
    try:
        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                # Fetch title + explanation to create the vector context
                cur.execute("""
                    SELECT id, title, structured_data->>'explanation' 
                    FROM knowledge_node 
                    WHERE id = ANY(%s)
                """, (node_ids,))
                
                rows = cur.fetchall()
                for row in rows:
                    node_id, title, content = row
                    full_text = f"{title}: {content or ''}"
                    
                    vector = generate_embedding(full_text)
                    if vector:
                        cur.execute(
                            "UPDATE knowledge_node SET embedding = %s WHERE id = %s",
                            (vector, node_id)
                        )
            conn.commit()
        print("✅ Embeddings updated successfully.")
    except Exception as e:
        print(f"❌ Background job failed: {e}")


# --- CONSTANTS ---
SIMILARITY_THRESHOLD_DUPLICATE = 0.88  # It's the same thing
SIMILARITY_THRESHOLD_MERGE = 0.82      # It's related, check for new details

def smart_merge_or_create(cur, parent_id, node_data, user_email):
    """
    Intelligent logic to Check, Match, and Insert/Merge.
    Now tracks created_by / updated_by.
    """
    title = node_data['title']
    narrative = node_data.get('narrative', '')
    new_struct = node_data.get('structured_data', {})
    slug = node_data.get('slug')
    node_type = node_data['node_type']

    # 1. Generate Embedding
    vector = generate_embedding(f"{title}: {narrative}")
    
    # Fallback to simple title match if OpenAI fails
    if not vector:
        cur.execute("SELECT id FROM knowledge_node WHERE title = %s AND parent_id = %s", (title, parent_id))
        res = cur.fetchone()
        if res: return str(res[0]), "exists_title_match"

    # 2. Vector Search against SIBLINGS
    cur.execute("""
        SELECT id, title, structured_data, 1 - (embedding <=> %s::vector) as similarity
        FROM knowledge_node 
        WHERE parent_id = %s 
        ORDER BY similarity DESC 
        LIMIT 1
    """, (vector, parent_id))
    
    match = cur.fetchone()
    
    # --- LOGIC DECISION TREE ---
    if match:
        existing_id, existing_title, existing_struct, similarity = match
        
        # SCENARIO A: DUPLICATE
        if similarity > SIMILARITY_THRESHOLD_DUPLICATE:
            return str(existing_id), f"skipped_duplicate_of_{existing_title}"

        # SCENARIO B: MERGE CANDIDATE
        if similarity > SIMILARITY_THRESHOLD_MERGE:
            merged_struct = existing_struct.copy()
            updated = False
            
            for key, val in new_struct.items():
                if key not in merged_struct:
                    merged_struct[key] = val
                    updated = True
                elif isinstance(val, list) and isinstance(merged_struct[key], list):
                    current_set = set(merged_struct[key])
                    for item in val:
                        if item not in current_set:
                            merged_struct[key].append(item)
                            updated = True
            
            if updated:
                # [UPDATE] Add updated_by and updated_at
                cur.execute("""
                    UPDATE knowledge_node 
                    SET structured_data = %s, 
                        updated_at = NOW(),
                        updated_by = %s
                    WHERE id = %s
                """, (json.dumps(merged_struct), user_email, existing_id))
                return str(existing_id), f"merged_new_info_into_{existing_title}"
            else:
                return str(existing_id), f"skipped_no_new_info_vs_{existing_title}"

    # SCENARIO C: NEW CONTENT (Insert)
    # [INSERT] Only set created_by
    cur.execute("""
        INSERT INTO knowledge_node (parent_id, title, node_type, slug, structured_data, embedding, created_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (
        parent_id, title, node_type, slug, 
        json.dumps({**new_struct, "explanation": narrative}),
        vector,
        user_email # created_by only
    ))
    
    return str(cur.fetchone()[0]), "created_new"


def resolve_parent_sync(cur, data, email):
    # Search by Title
    cur.execute("SELECT id FROM knowledge_node WHERE title ILIKE %s AND parent_id IS NULL", (data['title'],))
    existing = cur.fetchone()
    if existing: return str(existing[0])

    # Create New Domain Node
    vec = generate_embedding(f"{data['title']}: {data.get('narrative', '')}")
    cur.execute("""
        INSERT INTO knowledge_node (title, node_type, slug, structured_data, embedding, created_by)
        VALUES (%s, 'domain', %s, %s, %s, %s) RETURNING id
    """, (data['title'], data.get('slug'), json.dumps({"explanation": data.get('narrative')}), vec, email))
    return str(cur.fetchone()[0])

# --- Recursive Path Finder ---
def get_node_lineage_sql():
    """Returns SQL for a recursive CTE that builds the path string."""
    return """
    WITH RECURSIVE lineage AS (
        SELECT id, title, parent_id, CAST(title AS TEXT) as path
        FROM knowledge_node WHERE parent_id IS NULL
        UNION ALL
        SELECT n.id, n.title, n.parent_id, l.path || ' > ' || n.title
        FROM knowledge_node n
        JOIN lineage l ON n.parent_id = l.id
    )
    """

# --- Deep Existence Check ---
@app.route('/api/knowledge/check-existence', methods=['POST'])
def check_existence():
    data = request.json
    title = data.get('title', '').strip()
    narrative = data.get('narrative', '') or data.get('structured_data', {}).get('explanation', '')
    
    if not title: return jsonify({"error": "Title required"}), 400

    # Assistant uses "Breadcrumb Vector" strategy
    search_text = f"{title}: {narrative}"
    vector = generate_embedding(search_text)
    if not vector: return jsonify({"error": "Embedding failed"}), 500

    try:
        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                # Updated SQL inside check_existence route
                query = """
                WITH RECURSIVE lineage AS (
                    SELECT id, title, parent_id, CAST(title AS TEXT) as path
                    FROM knowledge_node WHERE parent_id IS NULL
                    UNION ALL
                    SELECT n.id, n.title, n.parent_id, l.path || ' > ' || n.title
                    FROM knowledge_node n JOIN lineage l ON n.parent_id = l.id
                )
                SELECT 
                    l.id, 
                    l.title, 
                    l.path, 
                    1 - (n.embedding <=> %s::vector) as similarity,
                    n.structured_data->>'explanation' as existing_explanation, -- <--- ADD THIS
                    n.parent_id
                FROM lineage l
                JOIN knowledge_node n ON l.id = n.id
                ORDER BY similarity DESC LIMIT 1
                """
                cur.execute(query, (vector,))
                match = cur.fetchone()

                if match:
                    node_id, m_title, m_path, score, m_explanation, m_parent_id = match
                    status = "new"
                    if score > SIMILARITY_THRESHOLD_DUPLICATE: status = "duplicate"
                    elif score > SIMILARITY_THRESHOLD_MERGE: status = "merge_candidate"

                    return jsonify({
                        "exists": score > 0.85,
                        "status": "merge_candidate" if score > 0.82 else "new",
                        "best_match": {
                            "id": str(node_id),
                            "parent_id": str(m_parent_id) if m_parent_id else None, # Give the AI the Parent ID
                            "title": m_title,
                            "path": m_path,
                            "similarity": round(float(score), 4),
                            "existing_explanation": m_explanation # The LLM reads this to decide
                        }
                    })
                return jsonify({"exists": False, "status": "new"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Optimized Storage ---
@app.route('/api/knowledge/store-structure', methods=['POST'])
def store_structure():
    data = request.json
    user_email = data.get('user_email')
    if not user_email: return jsonify({"success": False, "error": "Email required"}), 400

    id_map = {}
    report = []

    try:
        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                # 1. Resolve Parent (Synchronous Embedding for Immediate Visibility)
                p_node = data.get('parent_node')
                root_id = resolve_parent_sync(cur, p_node, user_email)
                id_map[p_node['temp_id']] = root_id

                # 2. Process Nodes (Supports target_existing_id)
                for node in data.get('nodes', []):
                    # Check if Assistant found an anchor
                    target_id = node.get('target_existing_id')
                    
                    if target_id:
                        # CASE: Update/Attach to existing node
                        real_id = target_id
                        status = "attached_to_existing"
                        # Update narrative if provided
                        if node.get('narrative'):
                            cur.execute("UPDATE knowledge_node SET updated_by=%s, updated_at=NOW() WHERE id=%s", (user_email, real_id))
                    else:
                        # CASE: Traditional Smart Merge or Create
                        parent_real_id = id_map.get(node.get('parent_temp_id'))
                        real_id, status = smart_merge_or_create(cur, parent_real_id, node, user_email)

                    id_map[node['temp_id']] = real_id
                    
                    # Handle Attachments
                    for att in node.get('attachments', []):
                        cur.execute("""
                            INSERT INTO knowledge_attachment (node_id, file_name, file_type, file_path)
                            VALUES (%s, %s, %s, %s)
                        """, (real_id, att['name'], att.get('type'), att['path']))

                    report.append({"title": node['title'], "status": status, "id": real_id})

                # Process Keywords & Relationships...
                # --- Step 3: Keywords ---
                keywords = data.get('keywords', [])
                for kw in keywords:
                    # Insert keyword and sync synonyms
                    cur.execute("""
                        INSERT INTO keyword (label, synonyms) 
                        VALUES (%s, %s)
                        ON CONFLICT (label) DO UPDATE SET synonyms = EXCLUDED.synonyms
                        RETURNING id
                    """, (kw['label'], json.dumps(kw.get('synonyms', []))))
                    keyword_id = cur.fetchone()[0]

                    # Link keyword to nodes via id_map
                    for temp_id in kw.get('node_temp_ids', []):
                        real_node_id = id_map.get(temp_id)
                        if real_node_id:
                            cur.execute("""
                                INSERT INTO node_keyword (node_id, keyword_id, weight)
                                VALUES (%s, %s, %s)
                                ON CONFLICT (node_id, keyword_id) 
                                DO UPDATE SET weight = EXCLUDED.weight
                            """, (real_node_id, keyword_id, kw.get('weight', 1.0)))

                # --- Step 4: Relationships (Knowledge Edges) ---
                relationships = data.get('relationships', [])
                for rel in relationships:
                    source_id = id_map.get(rel['source_temp_id'])
                    target_id = id_map.get(rel['target_temp_id'])
                    
                    if source_id and target_id:
                        # Check for existing edge to prevent duplicates
                        cur.execute("""
                            SELECT id FROM knowledge_edge 
                            WHERE source_node_id = %s AND target_node_id = %s AND relation_type = %s
                        """, (source_id, target_id, rel['relation_type']))
                        
                        if not cur.fetchone():
                            cur.execute("""
                                INSERT INTO knowledge_edge 
                                (source_node_id, target_node_id, relation_type, description, created_by)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (
                                source_id, 
                                target_id, 
                                rel['relation_type'], 
                                rel.get('description'), 
                                user_email # Audit tracking
                            ))

            conn.commit()

        # Update embeddings in background for new nodes
        threading.Thread(target=run_embedding_job, args=(list(id_map.values()),)).start()
        return jsonify({"success": True, "report": report, "root_id": root_id})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500




#========================================================================================================================
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'txt', 'csv', 'xlsx', 'docx', 'pptx', 'md', 'json'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_bytes_to_github(file_content_bytes, filename, folder_path="uploads"):
    """
    Uploads raw bytes to GitHub and returns the file path and raw URL.
    """
    try:
        # 1. Config
        token = os.getenv("GITHUB_TOKEN")
        repo_full_name = "STS-Engineer/Knowledge_Group" # Hardcoded based on your prompt
        branch = "main"
        
        if not token:
            return {"success": False, "error": "GITHUB_TOKEN not set"}

        # 2. Encode Content
        content_b64 = base64.b64encode(file_content_bytes).decode('utf-8')

        # 3. Construct Unique Path
        unique_filename = f"{uuid.uuid4().hex[:8]}_{int(time.time())}_{filename}"
        file_path_in_repo = f"{folder_path}/{unique_filename}"
        
        api_url = f"https://api.github.com/repos/{repo_full_name}/contents/{file_path_in_repo}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        
        payload = {
            "message": f"Upload attachment: {unique_filename}",
            "content": content_b64,
            "branch": branch
        }

        # 4. Send Request
        response = requests.put(api_url, headers=headers, json=payload, timeout=20)
        
        if response.status_code in [200, 201]:
            data = response.json()
            return {
                "success": True,
                "path": file_path_in_repo, 
                "raw_url": data.get('content', {}).get('download_url')
            }
        else:
            return {"success": False, "error": f"GitHub {response.status_code}: {response.text}"}

    except Exception as e:
        return {"success": False, "error": str(e)}
    

def download_from_openai_file_id(file_id):
    """
    Downloads file bytes from OpenAI Files API using file_id.
    """
    try:
        print(f"⬇️ Downloading from OpenAI file_id: {file_id}")
        response = client.files.content(file_id)
        return response.read()
    except Exception as e:
        raise RuntimeError(f"OpenAI download failed for {file_id}: {e}")


# --- Route: Upload from OpenAI References (No DB Insert) ---
@app.route('/api/knowledge/upload-attachment', methods=['POST'])
def upload_attachment():
    """
    1. Receives list of file references (OpenAI refs).
    2. Downloads content.
    3. Uploads to GitHub.
    4. Returns the GitHub paths (NO DB INSERTION).
    """
    data = request.get_json(silent=True) or {}
    refs = data.get('openaiFileIdRefs', [])

    if not refs:
        return jsonify({"message": "No openaiFileIdRefs provided"}), 400

    uploaded_results = []
    errors = []

    for file_ref in refs:
        try:
            # --- Normalize input ---
            if isinstance(file_ref, dict):
                file_id = file_ref.get("id")
                download_link = file_ref.get("download_link")
                original_name = file_ref.get("name") or "uploaded_file"
                mime_type = file_ref.get("mime_type")
            else:
                # If someone passed just a string, assume it's a file_id
                file_id = file_ref
                download_link = None
                original_name = "uploaded_file"
                mime_type = None

            if not file_id:
                errors.append("Missing file_id in file reference")
                continue

            # --- Download bytes (LINK → FILE_ID fallback) ---
            file_bytes = None

            if download_link:
                try:
                    print(f"⬇️ Trying download_link for {original_name}")
                    r = requests.get(download_link, timeout=15)
                    r.raise_for_status()
                    file_bytes = r.content
                except Exception as e:
                    print(f"⚠️ download_link failed, falling back to file_id: {e}")

            if file_bytes is None:
                file_bytes = download_from_openai_file_id(file_id)

            # --- Filename & validation ---
            filename_safe = secure_filename(original_name)
            if '.' not in filename_safe:
                filename_safe += ".bin"

            if not allowed_file(filename_safe):
                errors.append(f"{original_name}: File type not allowed")
                continue

            # --- Upload to GitHub ---
            gh_result = upload_bytes_to_github(
                file_bytes,
                filename_safe,
                folder_path="uploads"
            )

            if not gh_result["success"]:
                errors.append(f"{original_name}: {gh_result['error']}")
                continue

            uploaded_results.append({
                "original_name": original_name,
                "path": gh_result["path"],
                "url": gh_result["raw_url"]
            })

        except Exception as e:
            print(f"❌ Error processing {original_name}: {e}")
            errors.append(f"{original_name}: {str(e)}")


    if not uploaded_results and errors:
        return jsonify({"success": False, "message": "All uploads failed", "errors": errors}), 500

    return jsonify({
        "success": True,
        "message": f"Processed {len(uploaded_results)} files.",
        "files": uploaded_results,
        "errors": errors
    }), 200




@app.route("/health", methods=["GET"])
def health():
    return {"ok": True}

@app.route('/health-check', methods=['GET'])
def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": time.ctime(),
        "database": {"status": "unknown", "latency_ms": None},
        "environment": {
            "openai_api": "ok" if os.getenv("OPENAI_API_KEY") else "missing",
            "github_token": "ok" if os.getenv("GITHUB_TOKEN") else "missing"
        }
    }

    # Test Database connection and timing
    start_time = time.time()
    try:
        # We set a short timeout so the health check doesn't hang if Azure is down
        with psycopg.connect(DB_DSN, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                health_status["database"]["status"] = "connected"
                health_status["database"]["latency_ms"] = round((time.time() - start_time) * 1000, 2)
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["database"]["status"] = "disconnected"
        health_status["database"]["error"] = str(e)

    # If the DB is down, return 503 (Service Unavailable)
    code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), code




if __name__ == '__main__':
    app.run(debug=True, port=5000)
