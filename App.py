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

# --- Main Route: Store Structure ---
@app.route('/api/knowledge/store-structure', methods=['POST'])
def store_structure():
    data = request.json
    
    # 1. The ID Map: Converts GPT's 'temp_id' -> Postgres 'Real UUID'
    # Example: {'root_node': '550e8400-e29b...', 'child_1': '7d444840-...'}
    id_map = {}
    created_real_ids = []

    try:
        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                
                # --- Step A: Handle the Parent/Root ---
                parent_node = data.get('parent_node')
                parent_real_id = None

                if parent_node['action'] == 'link_existing':
                    # User selected an existing domain
                    parent_real_id = parent_node['id']
                    # Map the temp_id to this existing UUID
                    id_map[parent_node['temp_id']] = parent_real_id
                else:
                    # Create NEW Parent
                    cur.execute("""
                        INSERT INTO knowledge_node (title, node_type, structured_data) 
                        VALUES (%s, %s, %s) 
                        RETURNING id
                    """, (
                        parent_node['title'],
                        parent_node['node_type'],
                        json.dumps({"explanation": parent_node.get('narrative', '')})
                    ))
                    parent_real_id = str(cur.fetchone()[0])
                    # Save mapping
                    id_map[parent_node['temp_id']] = parent_real_id
                    created_real_ids.append(parent_real_id)

                # --- Step B: Create Children Nodes ---
                nodes = data.get('nodes', [])
                for node in nodes:
                    # Find the real parent UUID using our map
                    # If GPT said parent is 'root_node', we look up 'root_node' in id_map
                    temp_parent = node.get('parent_temp_id')
                    real_parent_uuid = id_map.get(temp_parent)

                    if not real_parent_uuid:
                        raise ValueError(f"Parent temp_id '{temp_parent}' not found in map. Order matters!")

                    # Prepare JSONB data
                    structured = node.get('structured_data', {})
                    structured['explanation'] = node.get('narrative')

                    cur.execute("""
                        INSERT INTO knowledge_node (parent_id, title, node_type, slug, structured_data)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        real_parent_uuid,
                        node['title'],
                        node['node_type'],
                        node.get('slug'),
                        json.dumps(structured)
                    ))
                    
                    real_child_id = str(cur.fetchone()[0])
                    # Update map for future sub-children or edges
                    id_map[node['temp_id']] = real_child_id
                    created_real_ids.append(real_child_id)

                # --- Step C: Keywords ---
                keywords = data.get('keywords', [])
                for kw in keywords:
                    # 1. Upsert Keyword (Create if not exists)
                    cur.execute("""
                        INSERT INTO keyword (label) VALUES (%s)
                        ON CONFLICT (label) DO UPDATE SET label = EXCLUDED.label
                        RETURNING id
                    """, (kw['label'],))
                    kw_id = cur.fetchone()[0]

                    # 2. Link to Nodes using ID Map
                    for node_temp_id in kw.get('node_temp_ids', []):
                        real_node_id = id_map.get(node_temp_id)
                        if real_node_id:
                            cur.execute("""
                                INSERT INTO node_keyword (node_id, keyword_id, weight)
                                VALUES (%s, %s, %s)
                                ON CONFLICT DO NOTHING
                            """, (real_node_id, kw_id, kw.get('weight', 1.0)))

                # --- Step D: Relationships (Edges) ---
                edges = data.get('relationships', [])
                for edge in edges:
                    source_real = id_map.get(edge['source_temp_id'])
                    target_real = id_map.get(edge['target_temp_id'])

                    if source_real and target_real:
                        cur.execute("""
                            INSERT INTO knowledge_edge (source_node_id, target_node_id, relation_type, description)
                            VALUES (%s, %s, %s, %s)
                        """, (source_real, target_real, edge['relation_type'], edge.get('description')))

            # Commit the transaction (All or Nothing)
            conn.commit()

        # --- Step E: Launch Background Job ---
        # We use a Thread so the API responds instantly, while Embeddings calculate in background
        thread = threading.Thread(target=run_embedding_job, args=(created_real_ids,))
        thread.start()

        return jsonify({
            "success": True,
            "message": "Knowledge structure integrated.",
            "root_id": parent_real_id,
            "nodes_created": len(created_real_ids)
        }), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)