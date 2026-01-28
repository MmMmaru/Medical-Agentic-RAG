import asyncio
import os
import shutil
import numpy as np
import sys

#为了能够导入模块，添加根目录到path
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DB.milvus_vectorDB import MilvusVectorStorage
# 尝试从 base 导入 DataChunk，如果文件有语法错误则使用 Mock
try:
    from base import DataChunk
except Exception as e:
    print(f"Could not import DataChunk: {e}, using MockDataChunk")
    class DataChunk:
        def __init__(self, chunk_id, doc_id, content, vector, file_path="test/path", chunk_index=0, metadata=None):
            self.chunk_id = chunk_id
            self.chunk_id = chunk_id
            self.doc_id = doc_id
            self.content = content
            self.vector = vector
            self.file_path = file_path
            self.chunk_index = chunk_index
            self.metadata = metadata or {}

async def main():
    workspace = "./test_milvus_workspace"
    embedding_dim = 8
    
    # 清理旧的测试环境
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.makedirs(workspace)
    
    print(f"=== Initializing MilvusVectorStorage in {workspace} ===")
    try:
        storage = MilvusVectorStorage(
            workspace=workspace,
            embedding_dim=embedding_dim
        )
    except ImportError:
        print("pymilvus not installed. Skipping test execution.")
        return
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # 1. 准备测试数据
    print("\n=== Preparing Test Data ===")
    doc_id_1 = "doc_1"
    doc_id_2 = "doc_2"
    
    # doc 1
    chunk1 = DataChunk(
        chunk_id="c1", 
        doc_id=doc_id_1, 
        content="This is the first chunk of document 1.",
        vector=np.random.rand(embedding_dim).tolist(),
        metadata={"author": "alice"}
    )
    # doc 1
    chunk2 = DataChunk(
        chunk_id="c2", 
        doc_id=doc_id_1, 
        content="This is the second chunk of document 1.",
        vector=np.random.rand(embedding_dim).tolist(),
        metadata={"author": "alice"}
    )
    # doc 2
    chunk3 = DataChunk(
        chunk_id="c3", 
        doc_id=doc_id_2, 
        content="This is a chunk from document 2.",
        vector=np.random.rand(embedding_dim).tolist(),
        metadata={"author": "bob"}
    )
    
    chunks = [chunk1, chunk2, chunk3]
    
    # 2. 测试 Upsert
    print("\n=== Testing Upsert ===")
    await storage.upsert(chunks)
    print("Upsert completed.")

    # 3. 测试 Search
    print("\n=== Testing Search ===")
    # 使用 chunk1 的向量进行搜索，期望得到 chunk1 最靠前
    query_vector = chunk1.vector
    results = await storage.search(query_vector, top_k=2)
    
    print(f"Search results (top 2): {len(results)}")
    for i, res in enumerate(results):
        print(f"Rank {i+1}: ID={res.chunk_id}, Score/Content={res.content[:20]}...")
        if i == 0:
            assert res.chunk_id == "c1", f"Expected top result to be c1, but got {res.chunk_id}"

    # 4. 测试 Delete
    print("\n=== Testing Delete by doc_id (doc_1) ===")
    await storage.delete_by_doc_id(doc_id_1)
    
    # 再次搜索，不应该找到 doc_1 的 chunk
    results_after_delete = await storage.search(query_vector, top_k=5)
    print(f"Search results after deleting doc_1: {len(results_after_delete)}")
    
    ids_remaining = [r.chunk_id for r in results_after_delete]
    print(f"Remaining IDs: {ids_remaining}")
    
    assert "c1" not in ids_remaining, "c1 should have been deleted"
    assert "c2" not in ids_remaining, "c2 should have been deleted"
    assert "c3" in ids_remaining, "c3 should still exist"

    # 5. 测试 Backup/Restore
    print("\n=== Testing Backup/Restore ===")
    backup_path = await storage.backup("test_backup")
    if backup_path:
        print(f"Backup created at: {backup_path}")
        
        # 模拟数据丢失（删除 c3）
        await storage.delete_by_doc_id(doc_id_2)
        res_empty = await storage.search(chunk3.vector, top_k=5)
        print(f"Results after deleting all: {len(res_empty)}")
        assert len(res_empty) == 0
        
        # 恢复
        await storage.restore("test_backup")
        res_restored = await storage.search(chunk3.vector, top_k=5)
        print(f"Results after restore: {len(res_restored)}")
        assert len(res_restored) >= 1
        assert res_restored[0].chunk_id == "c3"
    
    print("\n=== Test Completed Successfully ===")
    
    # 清理
    # storage.client.close() # MilvusClient might not have close method in strict sense or handled automatically
    shutil.rmtree(workspace)

if __name__ == "__main__":
    asyncio.run(main())
