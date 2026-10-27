"""Storage layer with two interchangeable backends:

  - LocalStore   — JSON files on disk. Zero infrastructure; what tests
                   and the default docker-compose use.
  - DynamoStore  — the design's production shape (§9, §14): one call
                   pattern, PK/SK keys, works against dynamodb-local in
                   the compose 'aws' profile or real DynamoDB on EC2.

Both expose the same tiny document API (put/get/scan/delete) so the
service layer and the ChatSessions store never know which is active.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

from .config import settings


class LocalStore:
    """A per-table JSON dict persisted to DATA_DIR — honest, inspectable."""

    def __init__(self, table: str):
        self.path = os.path.join(settings().DATA_DIR, f"{table}.json")
        os.makedirs(settings().DATA_DIR, exist_ok=True)
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self._items = json.load(f)

    def _flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._items, f, indent=1, default=str)
        os.replace(tmp, self.path)

    @staticmethod
    def _key(pk: str, sk: str) -> str:
        return f"{pk}||{sk}"

    def put(self, pk: str, sk: str, item: dict):
        with self._lock:
            self._items[self._key(pk, sk)] = {**item, "PK": pk, "SK": sk}
            self._flush()

    def get(self, pk: str, sk: str) -> Optional[dict]:
        return self._items.get(self._key(pk, sk))

    def scan(self, pk_prefix: str = "", sk_prefix: str = "") -> list[dict]:
        return [v for v in self._items.values()
                if v["PK"].startswith(pk_prefix) and v["SK"].startswith(sk_prefix)]

    def delete(self, pk: str, sk: str):
        with self._lock:
            self._items.pop(self._key(pk, sk), None)
            self._flush()


class DynamoStore:
    """Same API against DynamoDB (or dynamodb-local via DYNAMO_ENDPOINT)."""

    def __init__(self, table: str):
        import boto3  # imported only when this backend is selected
        kw: dict[str, Any] = {"region_name": settings().DYNAMO_REGION}
        if settings().DYNAMO_ENDPOINT:
            kw["endpoint_url"] = settings().DYNAMO_ENDPOINT
            kw["aws_access_key_id"] = "local"
            kw["aws_secret_access_key"] = "local"
        self.ddb = boto3.resource("dynamodb", **kw)
        self.table = self._ensure(table)

    def _ensure(self, name: str):
        try:
            t = self.ddb.Table(name)
            t.load()
            return t
        except Exception:
            t = self.ddb.create_table(
                TableName=name,
                KeySchema=[{"AttributeName": "PK", "KeyType": "HASH"},
                           {"AttributeName": "SK", "KeyType": "RANGE"}],
                AttributeDefinitions=[{"AttributeName": "PK", "AttributeType": "S"},
                                      {"AttributeName": "SK", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST")
            t.wait_until_exists()
            return t

    def put(self, pk: str, sk: str, item: dict):
        self.table.put_item(Item={**item, "PK": pk, "SK": sk})

    def get(self, pk: str, sk: str):
        return self.table.get_item(Key={"PK": pk, "SK": sk}).get("Item")

    def scan(self, pk_prefix: str = "", sk_prefix: str = ""):
        items = self.table.scan().get("Items", [])
        return [i for i in items
                if i["PK"].startswith(pk_prefix) and i["SK"].startswith(sk_prefix)]

    def delete(self, pk: str, sk: str):
        self.table.delete_item(Key={"PK": pk, "SK": sk})


_stores: dict[str, Any] = {}


def store(table: str):
    """Table factory honoring STORAGE_BACKEND."""
    if table not in _stores:
        if settings().STORAGE_BACKEND == "dynamodb":
            _stores[table] = DynamoStore(table)
        else:
            _stores[table] = LocalStore(table)
    return _stores[table]
