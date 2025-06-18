import requests

def download_all_pages(url, token):
    """Download all pages from a DokuWiki instance using JSON-RPC and a bearer token. Returns a list of page contents as strings."""
    session = requests.Session()
    session.headers.update({
        "Content-Type":"application/json",
        "Authorization":f"Bearer {token}"
    })
    def rpc(method, params=None):
        payload = {"jsonrpc": "2.0", "method": method, "id": 1}
        if params:
            payload["params"] = params
        resp = session.post(f"{url}/lib/exe/jsonrpc.php", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["result"]
    page_ids = rpc("dokuwiki.getPagelist", ["", {"depth": 0, "hash": False}])
    return [rpc("wiki.getPage", [p["id"]]) for p in page_ids]