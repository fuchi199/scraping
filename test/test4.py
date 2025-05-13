import aiohttp
import asyncio
import ssl
import pandas as pd
from datetime import datetime

ACCESS_TOKEN = "あなたのアクセストークン"
API_BASE = "https://cloud.uipath.com/<account>/<tenant>/orchestrator_/odata"

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# 今日のUTC 0:00〜現在
start_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"
end_time = datetime.utcnow().isoformat() + "Z"

async def get_folders(session):
    url = f"{API_BASE}/Folders"
    async with session.get(url, headers=HEADERS) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return [folder["Id"] for folder in data.get("value", [])]

async def fetch_logs_for_folder(session, folder_id):
    url = f"{API_BASE}/RobotLogs"
    params = {
        "$filter": f"Level eq 'Error' and Time ge {start_time} and Time le {end_time}",
        "$top": 1000,
        "$orderby": "ProcessName asc, Time desc"
    }

    logs = []
    headers_with_folder = HEADERS.copy()
    headers_with_folder["X-OrganizationUnitId"] = str(folder_id)

    while url:
        async with session.get(url, headers=headers_with_folder, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
            logs.extend(data.get("value", []))
            url = data.get("@odata.nextLink", None)
            params = None

    return logs

async def main():
    target_processes = []
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        folder_ids = await get_folders(session)
        tasks = [fetch_logs_for_folder(session, folder_id) for folder_id in folder_ids]
        all_results = await asyncio.gather(*tasks)

        all_logs = [
            log for sublist in all_results for log in sublist
            if (
                log.get("ProcessName") in target_processes and
                not any(kw in log.get("Message", "") for kw in ["太郎", "一郎"])
            )
            ]
        
        df = pd.DataFrame(all_logs)
        if not df.empty:
            print(df[["Time", "ProcessName", "Level", "Message"]])
        else:
            print("本日のエラーログはありません。")

if __name__ == "__main__":
    asyncio.run(main())
