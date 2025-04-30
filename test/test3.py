import ssl
import aiohttp
import asyncio
import pandas as pd

ACCESS_TOKEN = ""
API_BASE = ""

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async def fetch_queue_definitions(session):
    url = f"{API_BASE}/QueueDefinitions"
    async with session.get(url, headers=HEADERS) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data.get("value", [])

async def fetch_all_inprogress(session, queue_definition_id, name, en):
    url = f"{API_BASE}/QueueItems"
    params = {
        "$filter": f"QueueDefinitionId eq {queue_definition_id} and Status eq 'InProgress'",
        "$top": 100
    }

    items = []
    while url:
        async with session.get(url, headers=HEADERS, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
            values = data.get("value", [])
            items.extend(values)
            url = data.get("@odata.nextLink", None)
            params = None  # nextLinkにはすでにパラメータが入ってる

    return [{
        "name": name,
        "en": en,
        "item_url": itm['Id'],
        "title": itm['Status'],
        "price": itm.get('StartProcessingTime')
    } for itm in items]

async def main():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        queues = await fetch_queue_definitions(session)
        tasks = [fetch_all_inprogress(session, q['Id'], q['Name'], q.get('EntriesCount')) for q in queues]
        results = await asyncio.gather(*tasks)
        all_items = [item for sublist in results for item in sublist]

        # 重複除去（Idベースで）
        unique_items = {item["item_url"]: item for item in all_items}.values()

        df = pd.DataFrame(unique_items)
        print(df)

if __name__ == "__main__":
    asyncio.run(main())