import asyncio
import httpx

async def download_all_pages(base_url: str, token: str) -> list[dict[str, str]]:
    """
    Fetch all pages from a DokuWiki instance using JSON-RPC and a bearer token.
    Returns a list of {"id": str, "content": str} dicts.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    sem = asyncio.Semaphore(10)

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        async def rpc_call(method: str, params=None):
            payload = {"jsonrpc": "2.0", "method": method, "id": 1}
            if params:
                payload["params"] = params

            async with sem:
                resp = await client.post(f"{base_url}/lib/exe/jsonrpc.php", json=payload)
                resp.raise_for_status()
                data = resp.json()
                if "error" in data and data["error"]:
                    raise ValueError(f"JSON-RPC Error: {data['error']}")
                return data.get("result")
        async def get_page(meta: dict):
            page_id = meta.get("id")
            if not page_id:
                return None

            for attempt in range(3):
                try:
                    content = await rpc_call("wiki.getPage", [page_id])
                    if content:
                        return {"id": page_id, "content": content}
                except Exception:
                    pass
                await asyncio.sleep(1)
            return None

        page_list = await rpc_call("dokuwiki.getPagelist", ["", {"depth": 0, "hash": False}])
        tasks = [get_page(p) for p in page_list]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]
