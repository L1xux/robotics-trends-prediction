# scripts/test_chroma_dump.py
import os
import sys
from pathlib import Path

def main():
    try:
        import chromadb
    except ImportError:
        print("[!] chromadb가 설치되어 있지 않습니다. `pip install chromadb` 후 재시도하세요.")
        sys.exit(1)

    # === 설정 ===
    # 인덱싱 때 썼던 경로/컬렉션명과 맞춰주세요.
    # 예) db_path: ./chroma_db, collection_name: documents (또는 robotics_docs)
    db_path = os.environ.get("CHROMADB_PATH", "./data/chroma_db")
    preferred_names = [
        os.environ.get("CHROMADB_COLLECTION", "documents"),
        "documents",
        "robotics_docs",
    ]

    # === 클라이언트 연결 ===
    client = chromadb.PersistentClient(path=db_path)

    # 전체 컬렉션 나열
    cols = client.list_collections()
    print(f"[i] Chroma path = {Path(db_path).resolve()}")
    print(f"[i] Collections ({len(cols)}): {[c.name for c in cols]}")

    # 사용할 컬렉션 선택
    col = None
    for name in preferred_names:
        try:
            col = client.get_collection(name=name)
            print(f"[i] Using collection: {name}")
            break
        except Exception:
            continue

    if col is None:
        print("[!] 테스트할 컬렉션을 찾지 못했습니다. 위 목록에서 존재하는 이름으로 바꿔주세요.")
        sys.exit(1)

    # 총 문서(청크) 수
    try:
        count = col.count()
    except Exception as e:
        print(f"[!] count() 실패: {e}")
        count = None

    print(f"[i] Total records = {count}")

    # 샘플 몇 개 가져오기 (id / metadata / document)
    try:
        sample = col.get(
            limit=5,
            include=["metadatas", "documents"]  # embeddings는 크므로 기본 제외
        )
        ids = sample.get("ids", [])
        docs = sample.get("documents", [])
        metas = sample.get("metadatas", [])

        print(f"[i] Sample {len(ids)} items:")
        for i, (sid, doc, meta) in enumerate(zip(ids, docs, metas), 1):
            preview = (doc or "")[:160].replace("\n", " ")
            print(f"  #{i} id={sid}")
            print(f"     meta={meta}")
            print(f"     doc =\"{preview}{'…' if doc and len(doc)>160 else ''}\"")
    except Exception as e:
        print(f"[!] 샘플 조회 실패: {e}")

    # (옵션) 간단 키워드 포함 검색: where_document 사용 (버전에 따라 미지원일 수 있음)
    try:
        kw = "robot"  # 원하는 키워드로 수정
        print(f'[i] where_document contains "{kw}" 테스트')
        hit = col.get(
            where_document={"$contains": kw},
            limit=3,
            include=["documents", "metadatas"]
        )
        hids = hit.get("ids", [])
        if hids:
            print(f"    -> {len(hids)}개 매치")
            for i, (sid, doc) in enumerate(zip(hids, hit.get("documents", [])), 1):
                prev = (doc or "")[:120].replace("\n", " ")
                print(f"       - id={sid} doc=\"{prev}{'…' if doc and len(doc)>120 else ''}\"")
        else:
            print("    -> 매치 없음(정상일 수 있음)")
    except Exception as e:
        print(f"[!] where_document 검색 미지원/실패: {e}")

if __name__ == "__main__":
    main()
