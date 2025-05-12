import aiohttp
import asyncio
from datetime import datetime
import pandas as pd
import ssl

ACCESS_TOKEN = ""
API_BASE = "https://cloud.uipath.com/<account>/<tenant>/orchestrator_/odata"

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"
end_time = datetime.now().isoformat() + "Z"

async def fetch_today_logs(session):
    url = f"{API_BASE}/RobotLogs"
    params = {
        "$filter": f"Level eq 'Error' and Time ge {start_time} and Time le {end_time}",
        "$top": 1000,
        "$orderby": "ProcessName asc, Time desc"
    }

    all_logs = []

    while url:
        async with session.get(url, headers=HEADERS, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
            all_logs.extend(data.get("value", []))
            url = data.get("@odata.nextLink", None)
            params = None  # nextLinkにパラメータ含まれる

    return all_logs

async def main():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        logs = await fetch_today_logs(session)
        df = pd.DataFrame(logs)
        print(df[["Time", "ProcessName", "Level", "Message"]])

if __name__ == "__main__":
    asyncio.run(main())
