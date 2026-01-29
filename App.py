import os
import json
import threading
from flask import Flask, request, jsonify
import psycopg
from openai import OpenAI
from dotenv import load_dotenv

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
SIMILARITY_THRESHOLD_DUPLICATE = 0.95  # It's the same thing
SIMILARITY_THRESHOLD_MERGE = 0.85      # It's related, check for new details

def smart_merge_or_create(cur, parent_id, node_data):
    """
    Intelligent logic to Check, Match, and Insert/Merge.
    Returns: (node_id, status_message)
    """
    title = node_data['title']
    narrative = node_data.get('narrative', '')
    new_struct = node_data.get('structured_data', {})
    slug = node_data.get('slug')
    node_type = node_data['node_type']

    # 1. Generate Embedding for the NEW content immediately
    # We need this to compare with existing nodes
    vector = generate_embedding(f"{title}: {narrative}")
    if not vector:
        # Fallback to simple title match if OpenAI fails
        cur.execute("SELECT id FROM knowledge_node WHERE title = %s AND parent_id = %s", (title, parent_id))
        res = cur.fetchone()
        if res: return str(res[0]), "exists_title_match"
        # Else Insert (code below)

    # 2. Vector Search against SIBLINGS only (children of this parent)
    # We look for the most similar existing child
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
        
        # SCENARIO A: DUPLICATE (High Similarity)
        if similarity > SIMILARITY_THRESHOLD_DUPLICATE:
            return str(existing_id), f"skipped_duplicate_of_{existing_title}"

        # SCENARIO B: MERGE CANDIDATE (High-ish Similarity)
        # It talks about the same thing. Let's see if we have NEW data fields.
        if similarity > SIMILARITY_THRESHOLD_MERGE:
            # Check if new_struct has keys/values missing in existing_struct
            merged_struct = existing_struct.copy()
            updated = False
            
            # Simple merge logic for lists (e.g., Risks, Methods)
            for key, val in new_struct.items():
                if key not in merged_struct:
                    merged_struct[key] = val
                    updated = True
                elif isinstance(val, list) and isinstance(merged_struct[key], list):
                    # Append new items to the list (unique only)
                    current_set = set(merged_struct[key])
                    for item in val:
                        if item not in current_set:
                            merged_struct[key].append(item)
                            updated = True
            
            if updated:
                # Update the DB
                cur.execute("""
                    UPDATE knowledge_node 
                    SET structured_data = %s, updated_at = NOW()
                    WHERE id = %s
                """, (json.dumps(merged_struct), existing_id))
                return str(existing_id), f"merged_new_info_into_{existing_title}"
            else:
                return str(existing_id), f"skipped_no_new_info_vs_{existing_title}"

    # SCENARIO C: NEW CONTENT (Low similarity or no match)
    # Insert as a new sibling
    cur.execute("""
        INSERT INTO knowledge_node (parent_id, title, node_type, slug, structured_data, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (
        parent_id, title, node_type, slug, 
        json.dumps({**new_struct, "explanation": narrative}),
        vector # Save the vector immediately
    ))
    
    return str(cur.fetchone()[0]), "created_new"


def resolve_parent(cur, parent_node_data):
    """
    Robust Parent Resolver:
    1. If linking to ID -> Verify ID exists.
    2. If creating new -> Check if Title exists (Case Insensitive).
       - If yes -> Return EXISTING ID (Prevent Duplicate).
       - If no -> Create NEW Parent.
    """
    action = parent_node_data.get('action')
    title = parent_node_data.get('title', '').strip()
    
    # CASE 1: Explicit Link by ID
    if action == 'link_existing' and parent_node_data.get('id'):
        parent_id = parent_node_data.get('id')
        # Verification: Does this ID actually exist?
        cur.execute("SELECT id FROM knowledge_node WHERE id = %s", (parent_id,))
        if cur.fetchone():
            return str(parent_id)
        else:
            print(f"⚠️ Warning: Parent ID {parent_id} not found. Falling back to title search.")

    # CASE 2: Search by Title (Prevent Duplicates)
    # We use ILIKE for case-insensitive matching (e.g., "Finance" == "finance")
    cur.execute("""
        SELECT id FROM knowledge_node 
        WHERE title ILIKE %s AND node_type = 'domain'
    """, (title,))
    
    existing = cur.fetchone()
    
    if existing:
        print(f"✅ Found existing Parent '{title}'. Linking to ID: {existing[0]}")
        return str(existing[0])

    # CASE 3: Truly New Parent
    print(f"🆕 Creating NEW Parent Domain: '{title}'")
    
    # Generate embedding for the parent
    narrative = parent_node_data.get('narrative', '')
    slug = parent_node_data.get('slug')
    node_type = parent_node_data.get('node_type', 'domain')
    vector = generate_embedding(f"{title}: {narrative}")
    
    cur.execute("""
        INSERT INTO knowledge_node (title, node_type, slug, structured_data, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (
        title, 
        node_type, 
        slug, 
        json.dumps({"explanation": narrative}),
        vector
    ))
    
    return str(cur.fetchone()[0])

# --- Main Route: Store Structure ---
@app.route('/api/knowledge/store-structure', methods=['POST'])
def store_structure():
    data = request.json
    id_map = {}
    report_log = [] 
    
    try:
        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                
                # --- Step A: Handle the Parent/Root (Safe Resolve) ---
                p_node = data.get('parent_node')
                root_id = resolve_parent(cur, p_node) 
                
                # Save mapping so keywords/children can find it
                id_map[p_node['temp_id']] = root_id

                # --- Step B: Handle Children Nodes (Smart Merge) ---
                nodes = data.get('nodes', [])
                for node in nodes:
                    temp_parent = node.get('parent_temp_id')
                    real_parent_id = id_map.get(temp_parent)
                    
                    if not real_parent_id:
                        raise ValueError(f"Parent temp_id '{temp_parent}' not resolved.")

                    # Call Smart Function
                    real_child_id, status = smart_merge_or_create(
                        cur, 
                        real_parent_id, 
                        node
                    )
                    
                    id_map[node['temp_id']] = real_child_id
                    
                    # Log result
                    report_log.append({
                        "title": node['title'],
                        "status": status,
                        "id": real_child_id
                    })

                    # Handle Attachments (Optional)
                    attachments = node.get('attachments', [])
                    for att in attachments:
                        cur.execute("""
                            INSERT INTO knowledge_attachment (node_id, file_name, file_type, file_path)
                            VALUES (%s, %s, %s, %s)
                        """, (real_child_id, att['name'], att['type'], att['path']))

                # --- Step C: Keywords ---
                keywords = data.get('keywords', [])
                for kw in keywords:
                    # 1. Insert Keyword or Update Synonyms
                    cur.execute("""
                        INSERT INTO keyword (label, synonyms) 
                        VALUES (%s, %s)
                        ON CONFLICT (label) 
                        DO UPDATE SET synonyms = EXCLUDED.synonyms
                        RETURNING id
                    """, (
                        kw['label'], 
                        json.dumps(kw.get('synonyms', []))
                    ))
                    
                    keyword_id = cur.fetchone()[0]

                    # 2. Link to Nodes using ID Map
                    node_temps = kw.get('node_temp_ids', [])
                    weight = kw.get('weight', 1.0)
                    
                    for temp_id in node_temps:
                        real_node_id = id_map.get(temp_id)
                        
                        if real_node_id:
                            cur.execute("""
                                INSERT INTO node_keyword (node_id, keyword_id, weight)
                                VALUES (%s, %s, %s)
                                ON CONFLICT (node_id, keyword_id) 
                                DO UPDATE SET weight = EXCLUDED.weight
                            """, (real_node_id, keyword_id, weight))

                # --- Step D: Relationships (Edges) ---
                relationships = data.get('relationships', [])
                for rel in relationships:
                    source_id = id_map.get(rel['source_temp_id'])
                    target_id = id_map.get(rel['target_temp_id'])
                    
                    # Only create edge if both nodes exist in this batch/context
                    if source_id and target_id:
                        # Optional: Check for existing edge to prevent duplicates
                        cur.execute("""
                            SELECT id FROM knowledge_edge 
                            WHERE source_node_id = %s AND target_node_id = %s AND relation_type = %s
                        """, (source_id, target_id, rel['relation_type']))
                        
                        if not cur.fetchone():
                            cur.execute("""
                                INSERT INTO knowledge_edge (source_node_id, target_node_id, relation_type, description)
                                VALUES (%s, %s, %s, %s)
                            """, (
                                source_id, 
                                target_id, 
                                rel['relation_type'], 
                                rel.get('description')
                            ))

            conn.commit()

        # Launch background embeddings (Threaded)
        # Collect all IDs created or resolved
        all_affected_ids = list(id_map.values())
        thread = threading.Thread(target=run_embedding_job, args=(all_affected_ids,))
        thread.start()

        return jsonify({
            "success": True, 
            "report": report_log,
            "root_id": root_id
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)