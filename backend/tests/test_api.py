from __future__ import annotations

import io

from fastapi.testclient import TestClient


def test_health_reports_mode(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["llm"]["mode"] == "heuristic"


def test_demo_dataset_detail(client: TestClient):
    r = client.get("/api/datasets/demo")
    assert r.status_code == 200
    d = r.json()
    assert d["rows"] == 1500 and d["columns"] == 14
    assert "revenue" in d["numeric_columns"] and "order_date" in d["datetime_columns"]
    assert len(d["preview"]) == 8 and len(d["suggested_questions"]) >= 5


def test_upload_analyse_and_delete(client: TestClient):
    csv = "city,temp,rain\n" + "\n".join(
        f"c{i % 4},{10 + i % 7},{'yes' if i % 3 else 'no'}" for i in range(80)
    )
    r = client.post(
        "/api/datasets", files={"file": ("weather.csv", io.BytesIO(csv.encode()), "text/csv")}
    )
    assert r.status_code == 201, r.text
    ds = r.json()
    assert ds["rows"] == 80 and set(ds["numeric_columns"]) == {"temp"}

    r = client.get(f"/api/datasets/{ds['id']}/rows", params={"limit": 5, "offset": 2})
    assert r.status_code == 200 and len(r.json()["rows"]) == 5 and r.json()["total"] == 80

    r = client.post(f"/api/datasets/{ds['id']}/chat", json={"message": "average temp by city"})
    assert r.status_code == 200
    reply = r.json()
    assert reply["steps"][0]["tool"] == "group_aggregate" and reply["charts"]
    chart_url = reply["charts"][0]["url"]
    img = client.get(chart_url)
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"

    ids = [d["id"] for d in client.get("/api/datasets").json()]
    assert ds["id"] in ids and "demo" in ids
    assert client.delete(f"/api/datasets/{ds['id']}").status_code == 204
    assert client.get(f"/api/datasets/{ds['id']}").status_code == 404


def test_upload_validation(client: TestClient):
    r = client.post(
        "/api/datasets", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    )
    assert r.status_code == 400
    r = client.post("/api/datasets", files={"file": ("bad.csv", io.BytesIO(b"a,b\n"), "text/csv")})
    assert r.status_code == 422 and "no data rows" in r.json()["detail"]


def test_chat_history_session_and_report(client: TestClient):
    r = client.post("/api/datasets/demo/chat", json={"message": "hello"})
    sid = r.json()["session_id"]
    client.post(
        "/api/datasets/demo/chat", json={"message": "average revenue by region", "session_id": sid}
    )
    hist = client.get("/api/datasets/demo/history", params={"session_id": sid}).json()
    assert [h["role"] for h in hist] == ["user", "assistant", "user", "assistant"]
    assert hist[-1]["charts"] and hist[-1]["steps"][0]["tool"] == "group_aggregate"

    report = client.get("/api/datasets/demo/report", params={"session_id": sid})
    assert report.status_code == 200 and report.headers["content-type"].startswith("text/markdown")
    assert "# Analysis report" in report.text and "average revenue by region" in report.text


def test_chat_validation_and_missing_dataset(client: TestClient):
    assert client.post("/api/datasets/demo/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/datasets/nope/chat", json={"message": "hi"}).status_code == 404
    assert client.delete("/api/datasets/demo").status_code == 400


def test_artifact_path_traversal_is_blocked(client: TestClient):
    assert client.get("/api/artifacts/..%2F..%2Fetc%2Fpasswd").status_code in (404, 400)
    assert client.get("/api/artifacts/missing.png").status_code == 404
