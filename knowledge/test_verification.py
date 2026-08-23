import json
import hashlib
import re
import sys
import numpy as np
from pathlib import Path

# Resolve Paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
KNOWLEDGE_DIR = ROOT / "knowledge"

def get_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

def make_fingerprint(text: str) -> str:
    words = normalize_text(text).split()
    if len(words) < 8:
        words += ["null"] * (8 - len(words))
    return " ".join(words[:8])

def main():
    print("==================================================")
    # 1. Imports and Vectorizer Reference
    try:
        from utils.knowledge_retriever import hashing_query_embedding, KnowledgeRetriever
        print("1. IMPORTS & INSTANTIATION: PASS")
    except Exception as e:
        print(f"1. IMPORTS & INSTANTIATION: FAIL ({e})")
        return

    # 2. Same-Text Comparison Test
    print("\n==================================================")
    print("2. RUNNING SAME-TEXT VECTOR COMPARISON:")
    
    # Inline replication of the merge vectorizer to compare directly
    def merge_hashing_embedding(text: str, dimensions: int = 384) -> np.ndarray:
        words = " ".join(str(text).lower().split()).split()
        tokens = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        vector = np.zeros(dimensions, dtype=np.float32)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vector[value % dimensions] += 1.0 if (value >> 8) & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    test_texts = [
        "atrial fibrillation",
        "atrial fibrillation symptoms",
        "right bundle branch block",
        "hyperkalaemia ECG",
        "myocardial ischaemia"
    ]

    same_text_pass = True
    for text in test_texts:
        v_runtime = hashing_query_embedding(text)
        v_merge = merge_hashing_embedding(text)
        equal = np.allclose(v_runtime, v_merge, atol=1e-5)
        diff = np.max(np.abs(v_runtime - v_merge))
        print(f"Text: '{text}' -> np.allclose: {equal}, max_abs_diff: {diff:.6f}")
        if not equal:
            same_text_pass = False

    if same_text_pass:
        print("SAME-TEXT COMPARISON: PASS")
    else:
        print("SAME-TEXT COMPARISON: FAIL")

    # 3. Final Embedding Matrix & Invariants Check
    print("\n==================================================")
    print("3. VERIFYING EMBEDDINGS MATRIX INVARIANTS:")
    
    kb_path = KNOWLEDGE_DIR / "kb.json"
    id_list_path = KNOWLEDGE_DIR / "id_list.json"
    embeddings_path = KNOWLEDGE_DIR / "embeddings_kb.npy"

    if not (kb_path.is_file() and id_list_path.is_file() and embeddings_path.is_file()):
        print("Missing registry files in knowledge/. Running the retriever once to trigger migration...")
        try:
            retriever = KnowledgeRetriever()
        except Exception as e:
            print(f"Failed to auto-migrate: {e}")
            return

    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        with open(id_list_path, "r", encoding="utf-8") as f:
            id_list = json.load(f)
        embeddings = np.load(embeddings_path, allow_pickle=False)

        print(f"KB Chunks Count: {len(kb)}")
        print(f"IDs Count: {len(id_list)}")
        print(f"Embeddings Shape: {embeddings.shape}")
        print(f"Embeddings Dtype: {embeddings.dtype}")

        invariants_pass = (len(kb) == len(id_list) == embeddings.shape[0])
        print(f"Invariants Check (len(kb) == len(id) == shape[0]): {'PASS' if invariants_pass else 'FAIL'}")
    except Exception as e:
        print(f"Verification of invariants failed: {e}")
        return

    # 4. Stored Vector Reproduction
    print("\n==================================================")
    print("4. RUNNING STORED VECTOR REPRODUCTION TEST:")
    
    id_to_idx = {cid: idx for idx, cid in enumerate(id_list)}
    reproduction_pass = True
    
    # Test up to 3 old and 3 clinical chunks
    test_chunks = [
        "SCP-NDT-ECG_DIAGNOSTIC_CRITERIA",
        "SCP-NDT-SYMPTOMS",
        "SCP-NDT-DIFFERENTIAL_DIAGNOSIS",
        "af_008",
        "litfl_right-bundle-branch-block-rbbb-ecg-library_001",
        "litfl_pre-excitation-syndromes-ecg-library_003"
    ]

    for cid in test_chunks:
        # Find chunk
        chunk = next((c for c in kb if c["id"] == cid), None)
        if chunk is None:
            print(f"Chunk ID '{cid}' not found in current kb.json. Skipping.")
            continue
            
        row_idx = id_to_idx.get(cid)
        if row_idx is None:
            print(f"Chunk ID '{cid}' not found in id_list.json. Skipping.")
            reproduction_pass = False
            continue
            
        stored_vec = embeddings[row_idx]
        text_to_embed = chunk.get("evidence_summary") or chunk["text"]
        regen_vec = hashing_query_embedding(text_to_embed)
        
        equal = np.allclose(stored_vec, regen_vec, atol=1e-5)
        diff = np.max(np.abs(stored_vec - regen_vec))
        print(f"Chunk: {cid} -> np.allclose: {equal}, max_abs_diff: {diff:.6f}")
        if not equal:
            reproduction_pass = False

    if reproduction_pass:
        print("STORED VECTOR REPRODUCTION: PASS")
    else:
        print("STORED VECTOR REPRODUCTION: FAIL")

    # 5. Retrieval Quality Tests
    print("\n==================================================")
    print("5. RUNNING MOCK RETRIEVAL QUALITY TESTS:")
    try:
        retriever = KnowledgeRetriever()
        queries = [
            ("AF symptoms", "AFIB", "STTC"),
            ("RBBB meaning", "RBBB", "Conduction block"),
            ("1AVB meaning", "1AVB", "Conduction block"),
            ("atrial fibrillation treatment", "AFIB", "STTC"),
            ("inferior STEMI", "AMI", "Myocardial Infarction"),
            ("hyperkalaemia ECG", "NORM", "Normal ECG")
        ]
        
        for q_text, concept, family in queries:
            print(f"\nQuery: '{q_text}' (Concept: {concept}, Family: {family})")
            # Query the retriever using unified retrieve
            results = retriever.retrieve(
                question=q_text,
                query_concept=concept,
                query_family=family,
                top_k=3
            )
            # Support list or dict response
            chunks = results if isinstance(results, list) else results.get("all_chunks", [])
            
            for idx, res in enumerate(chunks[:3]):
                cid = res.get("id") or res.get("chunk_id")
                source = res.get("source_id") or "ESC"
                summary = str(res.get("evidence_summary") or res.get("text", ""))[:90]
                print(f"  [{idx+1}] ID: {cid} | Source: {source} | Snippet: {summary}...")
        print("\nRETRIEVAL QUALITY TESTS COMPLETED.")
    except Exception as e:
        print(f"Retrieval quality tests failed: {e}")

    print("==================================================")

if __name__ == "__main__":
    main()
