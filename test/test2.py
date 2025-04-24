import requests

headers = {
    "Authorization": f"Bearer {'アクセストークン'}",
    "Content-Type": "application/json"
}

# フォルダ一覧を取得
api_base = "https://cloud.uipath.com/{ACCOUNT_LOGICAL_NAME}/{SERVICE_NAME}/orchestrator_/odata"
url = f"{api_base}/Folders"
response = requests.get(url, headers=headers)
response.raise_for_status()
folders = response.json()["value"]
for f in folders:
    print(f"Folder ID: {f['Id']}, Name: {f['DisplayName']}")


# キュー
url = f"{api_base}/QueueDefinitions"
response2 = requests.get(url, headers=headers)
response2.raise_for_status()
data   = response2.json()
queues = data.get("value", [])
print(f"Queues retrieved: {len(queues)}")
for q in queues:
    print(f" - ID: {q['Id']}, Name: {q['Name']}, EntriesCount: {q.get('EntriesCount')}")


# トランザクション
queue_definition_id = 1092410 

url = f"{url}/QueueItems"
params = {
    "$filter": f"QueueDefinitionId%20eq%20{queue_definition_id}",
    "$top": 100
}

resp = requests.get(url, headers=headers, params=params)
resp.raise_for_status()

items = resp.json().get("value", [])

print(f"Retrieved {len(items)} transaction items for QueueDefinitionId={queue_definition_id}\n")
for itm in items:
    print("─── Transaction Item ───")
    print(f"Item ID:              {itm['Id']}")
    print(f"Status:               {itm['Status']}")
    print(f"Start Processing Time: {itm.get('StartProcessingTime')}")
    print(f"End Processing Time:   {itm.get('EndProcessingTime')}")
    print("SpecificContent:")
    for key, value in itm.get("SpecificContent", {}).items():
        print(f"  {key}: {value}")
    print()