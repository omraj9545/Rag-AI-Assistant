import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.database import init_db
import fitz

@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    await init_db()
    yield

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_features_pipeline(client, tmp_path):
    # 1. Create two temporary test PDFs
    
    # Paper 1
    pdf_path1 = tmp_path / "paper1.pdf"
    doc1 = fitz.open()
    page1 = doc1.new_page()
    page1.insert_text((50, 50), "Abstract. We introduce neural networks for computer vision. Introduction. Deep learning is useful.")
    doc1.save(str(pdf_path1))
    doc1.close()
    
    # Paper 2
    pdf_path2 = tmp_path / "paper2.pdf"
    doc2 = fitz.open()
    page2 = doc2.new_page()
    page2.insert_text((50, 50), "Abstract. We study transformers for natural language. Introduction. Attention is powerful.")
    doc2.save(str(pdf_path2))
    doc2.close()

    # 2. Upload both papers
    with open(pdf_path1, "rb") as f1:
        resp1 = await client.post("/papers/upload", files={"file": ("paper1.pdf", f1, "application/pdf")})
    assert resp1.status_code == 200
    paper1_id = resp1.json()["paper_id"]

    with open(pdf_path2, "rb") as f2:
        resp2 = await client.post("/papers/upload", files={"file": ("paper2.pdf", f2, "application/pdf")})
    assert resp2.status_code == 200
    paper2_id = resp2.json()["paper_id"]

    # 3. Test Map-Reduce Summarization (mocked LLM)
    mock_sum_resp = "OBJECTIVE: Test objective\nMETHODOLOGY: Test methodology\nKEY FINDINGS: - Finding 1"
    with patch("app.services.llm.LLMService.complete", new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = mock_sum_resp
        
        sum_resp = await client.get(f"/papers/{paper1_id}/summarize")
        assert sum_resp.status_code == 200
        assert sum_resp.json()["summary"] == mock_sum_resp
        
        # Verify complete was called (mapped for the chunks and reduced at the end)
        assert mock_complete.call_count >= 2  # At least 1 map call + 1 reduce call

    # 5. Test Cross-paper Semantic Search
    search_resp = await client.get("/search/?q=transformers")
    assert search_resp.status_code == 200
    data = search_resp.json()
    assert data["query"] == "transformers"
    assert len(data["results"]) > 0
    # verify enrichment contains paper fields
    assert "paper" in data["results"][0]
    assert "title" in data["results"][0]["paper"]

    # 6. Test Multi-paper Comparison (mocked LLM)
    mock_comp_resp = "Comparison table:\n- Paper 1: neural networks\n- Paper 2: transformers"
    with patch("app.services.llm.LLMService.complete", new_callable=AsyncMock) as mock_complete_comp:
        mock_complete_comp.return_value = mock_comp_resp
        
        comp_resp = await client.post(
            "/compare/",
            json={"paper_ids": [paper1_id, paper2_id], "aspect": "architecture"}
        )
        assert comp_resp.status_code == 200
        assert comp_resp.json()["comparison"] == mock_comp_resp
        assert len(comp_resp.json()["papers"]) == 2

    # 7. Clean up by deleting papers
    delete_resp1 = await client.delete(f"/papers/{paper1_id}")
    assert delete_resp1.status_code == 200
    delete_resp2 = await client.delete(f"/papers/{paper2_id}")
    assert delete_resp2.status_code == 200
