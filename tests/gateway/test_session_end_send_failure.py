"""A failed ``adapter.send`` while delivering plugin ``on_session_finalize`` text is a WARNING naming the session
and the error (it used to be a debug line nobody saw: card ab). No network: a fake adapter."""
import asyncio
import logging
from types import SimpleNamespace

from gateway.run_shutdown import GatewayShutdownMixin


class _Runner(GatewayShutdownMixin):
    def __init__(self, adapter):
        self._adapter = adapter

    def _delivery_adapter_for(self, source):
        return self._adapter

    def _thread_metadata_for_source(self, source):
        return None


def _source():
    return SimpleNamespace(chat_id="1000000001", platform=SimpleNamespace(value="telegram"))


def _deliver(adapter, caplog):
    caplog.set_level(logging.DEBUG, logger="gateway.run")
    asyncio.run(_Runner(adapter)._deliver_session_end_messages(
        ["Handed off"], source=_source(), session_key="agent:main:telegram:dm:42"))
    return [r for r in caplog.records if r.name == "gateway.run"]


def test_send_raising_logs_a_warning_with_session_and_exception(caplog):
    async def send(chat, msg, **kw):
        raise RuntimeError("telegram 502 bad gateway")
    records = _deliver(SimpleNamespace(send=send), caplog)
    assert [r.levelno for r in records] == [logging.WARNING]
    line = records[0].getMessage()
    assert line == ("Failed to send session-end plugin message (session agent:main:telegram:dm:42) "
                    "to telegram:1000000001: telegram 502 bad gateway")
    print(line)


def test_send_returning_failure_logs_a_warning_with_the_error(caplog):
    async def send(chat, msg, **kw):
        return SimpleNamespace(success=False, error="chat not found")
    records = _deliver(SimpleNamespace(send=send), caplog)
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "agent:main:telegram:dm:42" in records[0].getMessage() and "chat not found" in records[0].getMessage()


def test_delivered_send_logs_no_warning(caplog):
    async def send(chat, msg, **kw):
        return SimpleNamespace(success=True, message_id="1")
    assert _deliver(SimpleNamespace(send=send), caplog) == []
