from fastapi.testclient import TestClient

from app.main import app


def test_api_responses_include_private_security_headers():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "camera=()" in response.headers["permissions-policy"]
    assert "default-src 'none'" in response.headers["content-security-policy"]


def test_untrusted_host_is_rejected():
    client = TestClient(app)

    response = client.get("/health", headers={"Host": "attacker.invalid"})

    assert response.status_code == 400
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
