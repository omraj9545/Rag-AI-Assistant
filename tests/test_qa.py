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
async def test_qa_endpoint_with_mocked_llm(client, tmp_path):
    # 1. Create a temporary test PDF
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Abstract. This paper introduces the Transformer architecture. Introduction. It outperforms other architectures.")
    doc.save(str(pdf_path))
    doc.close()

    # 2. Upload and ingest the PDF
    with open(pdf_path, "rb") as f:
        upload_resp = await client.post("/papers/upload", files={"file": ("test.pdf", f, "application/pdf")})
    
    assert upload_resp.status_code == 200
    paper_data = upload_resp.json()
    paper_id = paper_data["paper_id"]

    # 3. Ask a question with mocked LLMService.complete
    mock_complete_resp = "The paper introduces the Transformer architecture [Excerpt 1]."
    
    with patch("app.services.llm.LLMService.complete", new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = mock_complete_resp
        
        qa_resp = await client.post(
            f"/papers/{paper_id}/ask",
            json={"question": "What is introduced in this paper?", "top_k": 3}
        )
        
        assert qa_resp.status_code == 200
        data = qa_resp.json()
        assert data["question"] == "What is introduced in this paper?"
        assert data["answer"] == mock_complete_resp
        assert data["paper_id"] == paper_id
        assert len(data["sources"]) > 0
        assert data["sources"][0]["excerpt_num"] == 1
        assert "Transformer" in data["sources"][0]["text"]
        
        # Verify LLM was invoked with appropriate prompt constraints
        mock_complete.assert_called_once()
        args, _ = mock_complete.call_args
        system_prompt = args[0]
        user_prompt = args[1]
        assert "expert research assistant" in system_prompt
        assert "Transformer" in user_prompt

    # 4. Clean up by deleting the paper
    delete_resp = await client.delete(f"/papers/{paper_id}")
    assert delete_resp.status_code == 200
