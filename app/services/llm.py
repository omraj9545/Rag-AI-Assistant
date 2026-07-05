import httpx
from app.config import settings

class LLMService:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider.lower() if provider else settings.llm_provider.lower()
        self.ollama_host = settings.ollama_host
        self.model = model if model else settings.llm_model
        self.groq_api_key = settings.groq_api_key
        self.gemini_api_key = settings.gemini_api_key
        self.qwen_api_key = settings.qwen_api_key
        self.openrouter_api_key = settings.openrouter_api_key

    async def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if self.provider == 'ollama':
            url = f"{self.ollama_host.rstrip('/')}/api/chat"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "stream": False,
                "options": {
                    "num_predict": max_tokens
                }
            }
            # Set a high timeout because local inference can take some time
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                return result["message"]["content"]

        elif self.provider == 'groq':
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY is not set in environment or config.")
            
            # Map default local model to a fast Groq model if the user didn't override it
            model_name = self.model if self.model != 'qwen3:8b' else 'llama-3.3-70b-specdec'
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "max_tokens": max_tokens
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]

        elif self.provider == 'gemini':
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not set in environment or config.")
            
            # Map default local model to the high-quota Gemini 2.5 Flash Lite model if not overridden
            model_name = self.model if self.model != 'qwen3:8b' else 'gemini-2.5-flash-lite'
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": user}
                        ]
                    }
                ],
                "systemInstruction": {
                    "parts": [
                        {"text": system}
                    ]
                },
                "generationConfig": {}
            }
            import asyncio
            retries = 3
            backoff = 1.0
            for attempt in range(retries):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.post(url, headers=headers, json=payload)
                        response.raise_for_status()
                        result = response.json()
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (500, 502, 503, 504) and attempt < retries - 1:
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise e
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    if attempt < retries - 1:
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise e

        elif self.provider == 'qwen':
            if not self.qwen_api_key:
                raise ValueError("QWEN_API_KEY is not set in environment or config.")
            
            # Default model name mapping for DashScope (defaults to qwen-plus)
            model_name = self.model if self.model != 'qwen3:8b' else 'qwen-plus'
            url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.qwen_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "max_tokens": max_tokens
            }
            import asyncio
            retries = 3
            backoff = 1.0
            for attempt in range(retries):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.post(url, headers=headers, json=payload)
                        response.raise_for_status()
                        result = response.json()
                        return result["choices"][0]["message"]["content"]
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (500, 502, 503, 504) and attempt < retries - 1:
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise e
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    if attempt < retries - 1:
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise e

        elif self.provider == 'openrouter':
            if not self.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY is not set in environment or config.")
            
            # Default model name mapping for OpenRouter (defaults to stable Qwen 2.5 72B model)
            model_name = self.model if self.model != 'qwen3:8b' else 'qwen/qwen-2.5-72b-instruct'
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "AI Research Assistant",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "max_tokens": max_tokens
            }
            import asyncio
            retries = 3
            backoff = 1.0
            for attempt in range(retries):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.post(url, headers=headers, json=payload)
                        response.raise_for_status()
                        result = response.json()
                        if "error" in result:
                            # Raise custom error message from OpenRouter
                            raise httpx.HTTPStatusError(
                                message=result["error"].get("message", "OpenRouter API Error"),
                                request=response.request,
                                response=response
                            )
                        return result["choices"][0]["message"]["content"]
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (500, 502, 503, 504) and attempt < retries - 1:
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise e
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    if attempt < retries - 1:
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise e

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
