import time
from pathlib import Path
import httpx

client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0)

# 1. Check health
r0 = client.get("/api/v1/system/status")
print("0. Health Status:", r0.json()["status"], "| DB:", r0.json()["database"], "| Model:", r0.json()["face_model"])

# 2. Register
uid = f"demo_user_{int(time.time())}"
r1 = client.post("/api/v1/users/register", json={
    "user_id": uid,
    "full_name": "Demo Real Subject",
    "email": f"{uid}@example.com"
})
print("1. Registration status:", r1.status_code, r1.json()["user_id"])

# 3. Face Enroll with real gallery clip G_F 1.mp4
gallery_path = Path(r"C:\Users\sahit\Downloads\MSU_AVIS\MSU_AVIS\Gallery\face\G_F 1.mp4")
with open(gallery_path, "rb") as f:
    r2 = client.post(
        "/api/v1/face/enroll",
        data={"user_id": uid},
        files={"file": ("G_F 1.mp4", f, "video/mp4")}
    )
data2 = r2.json()
print("2. Enrollment status:", r2.status_code, "Dim:", data2["embedding_dim"], "Quality:", f"{data2['quality_score']*100:.1f}%")

# 4. Face Verify Genuine probe clip S01_V1/clip_01.mp4
probe_path = Path(r"C:\Users\sahit\Downloads\MSU_AVIS\MSU_AVIS\Probe_clips\S01_V1\clip_01.mp4")
with open(probe_path, "rb") as f:
    r3 = client.post(
        "/api/v1/face/verify",
        data={"user_id": uid},
        files={"file": ("clip_01.mp4", f, "video/mp4")}
    )
data3 = r3.json()
print(f"3. GENUINE Verification: Sim={data3['similarity_score']:.4f} | Thresh={data3['threshold']} | Decision={data3['decision']} | Risk={data3['risk_level']}")
assert data3["decision"] == "ACCEPT", f"Expected ACCEPT for genuine, got {data3['decision']}"

# 5. Face Verify Impostor probe clip S02_V1/clip_01.mp4
impostor_path = Path(r"C:\Users\sahit\Downloads\MSU_AVIS\MSU_AVIS\Probe_clips\S02_V1\clip_01.mp4")
with open(impostor_path, "rb") as f:
    r4 = client.post(
        "/api/v1/face/verify",
        data={"user_id": uid},
        files={"file": ("clip_01.mp4", f, "video/mp4")}
    )
data4 = r4.json()
print(f"4. IMPOSTOR Verification: Sim={data4['similarity_score']:.4f} | Thresh={data4['threshold']} | Decision={data4['decision']} | Risk={data4['risk_level']}")
assert data4["decision"] in ["REJECT", "STEP_UP"], f"Expected REJECT for impostor, got {data4['decision']}"

# 6. Check Audit Logs in MySQL
r5 = client.get("/api/v1/auth/logs")
logs = r5.json()
print(f"5. Total Audit Logs in DB: {len(logs)} | Latest attempt: {logs[0]['attempt_id']} -> {logs[0]['decision']}")

print("\n>>> ALL END-TO-END HTTP TESTS PASSED WITH REAL MSU-AVIS DATA! <<<")
