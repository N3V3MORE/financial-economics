"""WRDS login selection and sanitized failures; no real login or network calls."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from financial_economics.datasets import credentials as wrds_credentials
from financial_economics.datasets.sources import crsp as wrds_source


@pytest.fixture
def saved_login(monkeypatch):
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    reader = Mock(return_value=("invented_user", "invented_password"))
    monkeypatch.setattr(wrds_credentials, "get_saved_login", reader)
    connection = Mock()
    connect = Mock(return_value=connection)
    monkeypatch.setitem(sys.modules, "psycopg2", SimpleNamespace(connect=connect))
    return reader, connect, connection


def test_windows_login_is_reused_in_memory_for_one_readonly_connection(saved_login, capsys):
    reader, connect, connection = saved_login
    assert wrds_source.connect_wrds() is connection
    reader.assert_called_once_with()
    connect.assert_called_once_with(
        host=wrds_source.HOST, port=wrds_source.PORT, dbname="wrds", sslmode="require",
        connect_timeout=120, application_name="financial-economics",
        options="-c statement_timeout=60000", user="invented_user", password="invented_password")
    connection.set_session.assert_called_once_with(readonly=True, isolation_level="REPEATABLE READ")
    assert capsys.readouterr() == ("", "")


def test_absent_windows_login_keeps_pgpass_fallback(saved_login, monkeypatch):
    reader, connect, _ = saved_login
    reader.return_value = None
    monkeypatch.setattr(wrds_source, "_saved_username", lambda: "pgpass_user")
    wrds_source.connect_wrds()
    assert connect.call_args.kwargs["user"] == "pgpass_user"
    assert "password" not in connect.call_args.kwargs


def test_conflicting_explicit_username_prevents_wrong_account_login(saved_login, monkeypatch):
    _, connect, _ = saved_login
    monkeypatch.setenv("WRDS_USERNAME", "different_user")
    with pytest.raises(wrds_source.WRDSError, match="SAVED_LOGIN_MISMATCH"):
        wrds_source.connect_wrds()
    connect.assert_not_called()


def test_credential_manager_error_is_sanitized_without_fallback(saved_login):
    reader, connect, _ = saved_login
    reader.side_effect = wrds_credentials.CredentialError("invented_password")
    with pytest.raises(wrds_source.WRDSError, match="SAVED_LOGIN_FAILED") as caught:
        wrds_source.connect_wrds()
    assert "invented_password" not in str(caught.value) and caught.value.__suppress_context__
    connect.assert_not_called()


@pytest.mark.parametrize("failure", ["connect", "session", "close"])
def test_provider_failures_hide_credentials_and_never_retry(saved_login, failure):
    _, connect, connection = saved_login
    if failure == "connect":
        connect.side_effect = RuntimeError("invented_password")
    else:
        connection.set_session.side_effect = RuntimeError("invented_password")
        if failure == "close":
            connection.close.side_effect = RuntimeError("invented_password")
    with pytest.raises(wrds_source.WRDSError, match="CONNECTION_FAILED") as caught:
        wrds_source.connect_wrds()
    assert "invented_password" not in str(caught.value) and caught.value.__suppress_context__
    connect.assert_called_once()
    if failure != "connect":
        connection.close.assert_called_once()
