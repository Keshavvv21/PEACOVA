"""Tests for /api/admin/router (app.routers.admin_router_config) — RBAC
on both flags (ROUTER_SERVICE_ENABLED, the strategy toggle) and on the
"GENERIC ARCHITECTURE" comparison-arm proxy endpoints. The local
service-config flag needs no network; the strategy/generic-architecture
proxies talk to ak-router, which isn't running in this test process —
those assert a clean 502 (unreachable) for a role that's ALLOWED
through, and 403 (never even attempted) for one that isn't, so RBAC is
verified independently of ak-router actually being up.
"""

from tests.conftest import auth


def test_member_cannot_touch_any_router_admin_endpoint(client, member_token):
    for method, path in [
        ("get", "/api/admin/router/service-config"),
        ("patch", "/api/admin/router/service-config"),
        ("get", "/api/admin/router/strategy"),
        ("patch", "/api/admin/router/strategy"),
        ("post", "/api/admin/router/generic-architecture/route"),
        ("post", "/api/admin/router/generic-architecture/escalate"),
        ("post", "/api/admin/router/generic-architecture/outcome"),
        ("get", "/api/admin/router/routing-comparison"),
        ("get", "/api/admin/router/routing-comparison/report.pdf"),
    ]:
        kwargs = {"headers": auth(member_token)}
        if method != "get":
            kwargs["json"] = {}
        r = getattr(client, method)(path, **kwargs)
        assert r.status_code == 403, f"{method.upper()} {path} should be 403 for Member"


def test_admin_can_view_but_not_flip_service_config(client, admin_token):
    r = client.get("/api/admin/router/service-config", headers=auth(admin_token))
    assert r.status_code == 200
    assert r.json()["enabled"] is False  # default per router_service_config.py

    r = client.patch("/api/admin/router/service-config", json={"enabled": True},
                     headers=auth(admin_token))
    assert r.status_code == 403


def test_admin_can_view_strategy_but_not_patch_it(client, admin_token):
    # ak-router isn't running in this test process — a clean 502
    # ("unreachable"), not a crash, confirms the request actually got
    # past RBAC and failed only on the network hop.
    r = client.get("/api/admin/router/strategy", headers=auth(admin_token))
    assert r.status_code == 502

    r = client.patch("/api/admin/router/strategy", json={"strategy": "deterministic"},
                     headers=auth(admin_token))
    assert r.status_code == 403


def test_executive_toggles_service_config(client, exec_token):
    r = client.patch("/api/admin/router/service-config", json={"enabled": True},
                     headers=auth(exec_token))
    assert r.status_code == 200
    assert r.json()["enabled"] is True

    # Reset — other tests in this suite assume the default (disabled).
    r = client.patch("/api/admin/router/service-config", json={"enabled": False},
                     headers=auth(exec_token))
    assert r.status_code == 200
    assert r.json()["enabled"] is False


def test_executive_strategy_and_generic_architecture_reach_the_network_hop(client, exec_token):
    # Same reasoning as the admin-role strategy test above: Executive is
    # ALLOWED, so a 502 here proves RBAC let it through and it failed
    # only because ak-router isn't running in this test process.
    r = client.patch("/api/admin/router/strategy", json={"strategy": "deterministic"},
                     headers=auth(exec_token))
    assert r.status_code == 502

    r = client.post("/api/admin/router/generic-architecture/route", headers=auth(exec_token))
    assert r.status_code == 502

    r = client.post("/api/admin/router/generic-architecture/escalate", headers=auth(exec_token))
    assert r.status_code == 502

    r = client.post("/api/admin/router/generic-architecture/outcome",
                    json={"model_name": "openai/glm-5.3-flash", "tier": "default", "ok": True},
                    headers=auth(exec_token))
    assert r.status_code == 502


def test_patch_strategy_rejects_an_unknown_strategy(client, exec_token):
    r = client.patch("/api/admin/router/strategy", json={"strategy": "not-a-real-strategy"},
                     headers=auth(exec_token))
    assert r.status_code == 422


def test_admin_can_view_routing_comparison_and_report_but_hit_the_network_hop(client, admin_token):
    # Same reasoning as the strategy/generic-architecture tests above:
    # Admin is allowed through RBAC (it covers Senior Manager too, via
    # auth.py's _REAL_ROLE_MAP), so a clean 502 here proves it got past
    # the role check and failed only because ak-router isn't running in
    # this test process.
    r = client.get("/api/admin/router/routing-comparison", headers=auth(admin_token))
    assert r.status_code == 502

    r = client.get("/api/admin/router/routing-comparison/report.pdf", headers=auth(admin_token))
    assert r.status_code == 502


def test_routing_comparison_rejects_an_unknown_range(client, admin_token):
    r = client.get("/api/admin/router/routing-comparison", params={"range": "last-decade"},
                   headers=auth(admin_token))
    assert r.status_code == 422
