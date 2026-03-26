from __future__ import annotations
import os
import uuid
import logging
import time
from pathlib import Path
from threading import Timer
from flask import Flask, jsonify, request
from openai import OpenAI
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = BASE_DIR / "static" / "outputs"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
MAX_FILE_SIZE_MB = 20
TEMP_FILE_TTL_SECONDS = 1200  # 20 minutes

# Client OpenAI — échoue tôt si la clé est absente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY manquant dans .env")
client = OpenAI(api_key=OPENAI_API_KEY)

# =============================================================================
# UTILITAIRES
# =============================================================================

def delete_temp_file(filepath: str) -> None:
    """Supprime un fichier temporaire après expiration du TTL."""
    try:
        path = Path(filepath)
        if path.exists():
            path.unlink()
            logging.info(f"🗑️  Nettoyage : {path.name} supprimé (TTL {TEMP_FILE_TTL_SECONDS}s).")
    except Exception as e:
        logging.error(f"⚠️  Erreur nettoyage {filepath} : {e}")


def schedule_deletion(filepath: str, delay: int = TEMP_FILE_TTL_SECONDS) -> None:
    """Programme la suppression différée d'un fichier (daemon=True pour ne pas bloquer l'arrêt)."""
    t = Timer(delay, delete_temp_file, args=[filepath])
    t.daemon = True
    t.start()


def is_allowed_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def save_bytes_to_output(content: bytes, original_name: str) -> Path:
    """Génère un chemin unique et écrit les octets sur le disque."""
    safe_name = secure_filename(original_name) or "upload.png"
    unique_name = f"tmp_{uuid.uuid4().hex[:8]}_{safe_name}"
    filepath = OUTPUT_BASE / unique_name
    filepath.write_bytes(content)
    logging.info(f"💾 Fichier sauvegardé : {filepath}")
    return filepath

# =============================================================================
# TRAITEMENT MICROGRAPHIE (À REMPLACER PAR VOTRE LOGIQUE RÉELLE)
# =============================================================================

def process_image(filepath: Path) -> dict:
    """
    Remplacez cette fonction par votre pipeline OCR / PaddleOCR / détection.
    Doit retourner un dict sérialisable en JSON.
    """
    # Exemple : return detect_and_crop(str(filepath))
    return {
        "status": "simulation",
        "message": "Insérez ici votre logique OCR / Micrographie",
        "processed_file": filepath.name,
    }

# =============================================================================
# ENDPOINT PRINCIPAL : UPLOAD AND SEARCH
# =============================================================================

@app.route("/api/upload-and-search", methods=["POST"])
def upload_and_search():
    """
    Accepte :
      - Un JSON avec `openaiFileIdRefs` (liste d'objets {id, name} ou de strings)
      - Un formulaire multipart avec un champ `file`

    Flux :
      1. Récupération des octets (OpenAI Pull ou upload direct)
      2. Validation de l'extension
      3. Sauvegarde temporaire + planification de la suppression
      4. Traitement Micrographie
      5. Retour du résultat
    """
    saved_filepath: Path | None = None
    original_name = "upload.png"

    # ------------------------------------------------------------------
    # ÉTAPE 1 — SOURCE DES DONNÉES
    # ------------------------------------------------------------------
    data = request.get_json(silent=True) or {}
    refs = data.get("openaiFileIdRefs", [])

    if refs:
        # ---- CAS A : référence OpenAI ----
        ref = refs[0]
        if isinstance(ref, dict):
            file_id = ref.get("id", "")
            original_name = ref.get("name", "openai_upload.png")
        else:
            file_id = str(ref)

        if not file_id:
            return jsonify({"success": False, "error": "file_id vide dans openaiFileIdRefs"}), 400

        if not is_allowed_extension(original_name):
            return jsonify({"success": False, "error": f"Extension non autorisée : {original_name}"}), 415

        try:
            logging.info(f"📥 Pull OpenAI → file_id={file_id}")
            content = client.files.content(file_id).read()

            # Vérification de la taille
            size_mb = len(content) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                return jsonify({"success": False, "error": f"Fichier trop grand ({size_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB)"}), 413

            saved_filepath = save_bytes_to_output(content, original_name)

        except Exception as e:
            logging.error(f"❌ Erreur OpenAI Pull : {e}")
            return jsonify({"success": False, "error": f"Échec récupération OpenAI : {str(e)}"}), 502

    elif "file" in request.files:
        # ---- CAS B : upload multipart classique ----
        file = request.files["file"]
        original_name = file.filename or "upload.png"

        if not is_allowed_extension(original_name):
            return jsonify({"success": False, "error": f"Extension non autorisée : {original_name}"}), 415

        content = file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return jsonify({"success": False, "error": f"Fichier trop grand ({size_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB)"}), 413

        saved_filepath = save_bytes_to_output(content, original_name)

    else:
        return jsonify({"success": False, "error": "Aucun fichier ni openaiFileIdRefs fourni"}), 400

    # ------------------------------------------------------------------
    # ÉTAPE 2 — NETTOYAGE PROGRAMMÉ
    # ------------------------------------------------------------------
    schedule_deletion(str(saved_filepath))

    # ------------------------------------------------------------------
    # ÉTAPE 3 — TRAITEMENT
    # ------------------------------------------------------------------
    try:
        result = process_image(saved_filepath)
    except Exception as e:
        logging.error(f"❌ Erreur traitement image : {e}")
        return jsonify({"success": False, "error": f"Erreur analyse : {str(e)}"}), 500

    # ------------------------------------------------------------------
    # ÉTAPE 4 — RÉPONSE
    # ------------------------------------------------------------------
    return jsonify({
        "success": True,
        "filename": original_name,
        "expires_in_seconds": TEMP_FILE_TTL_SECONDS,
        "result": result,
    }), 200

# =============================================================================
# ROUTES ANNEXES
# =============================================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "Micrographie IA"}), 200


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
