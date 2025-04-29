import ssl
import aiohttp
import asyncio
import nest_asyncio
import pandas as pd

ACCESS_TOKEN = "rt_F7708875CF6BDC12FB526F3A76B48FBB6B7F19CD91BA1544424D0000B4CEE729-1"
API_BASE = "https://cloud.uipath.com/qkwjazie/DefaultTenant/orchestrator_/odata"

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

async def fetch_inprogress_transactions(session, queue_definition_id, name, en):
    url = f"{API_BASE}/QueueItems"
    params = {
        "$filter": f"QueueDefinitionId eq {queue_definition_id} and Status eq 'InProgress'",
        "$top": 100
    }
    async with session.get(url, headers=HEADERS, params=params) as resp:
        resp.raise_for_status()
        items = await resp.json()
        items = items.get("value", [])
        
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
        tasks = []
        for q in queues:
            tasks.append(fetch_inprogress_transactions(session, q['Id'], q['Name'], q.get('EntriesCount')))
        results = await asyncio.gather(*tasks)
        # フラットにまとめる
        all_items = [item for sublist in results for item in sublist]
        df = pd.DataFrame(all_items)
        print(df)

if __name__ == "__main__":
    asyncio.run(main())