"""Mock native credential APIs; never access real credentials or display dialogs."""

import ctypes
import traceback
from types import SimpleNamespace

import pytest

from financial_economics import wrds_credentials as credentials


USERNAME = "synthetic_用户🦉"
PASSWORD = "synthetic_密码🔑"


def _text(pointer, capacity=514):
    length = credentials._wide_length(pointer, capacity)
    return ctypes.string_at(pointer, length * 2).decode("utf-16-le")


@pytest.fixture(autouse=True)
def forbid_real_native_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not load actual Windows credential APIs")
    monkeypatch.setattr(credentials.ctypes, "WinDLL", forbidden, raising=False)


class FakeAPI:
    def __init__(self):
        self.read_calls = self.free_calls = self.write_calls = self.prompt_calls = 0
        self.read_success, self.write_success = True, True
        self.error, self.prompt_status = 0, 0
        self.username, self.password = USERNAME, PASSWORD
        self.username_buffer = credentials._wide_buffer(self.username)
        raw = self.password.encode("utf-16-le")
        self.blob = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
        self.saved = credentials._CREDENTIALW()
        self.saved.Type = 1
        self.saved.UserName = self.username_buffer
        self.saved.CredentialBlobSize = len(raw)
        self.saved.CredentialBlob = self.blob
        self.raise_read = self.raise_free = self.raise_prompt = self.raise_write = False

    def read(self, target, kind, flags, output):
        self.read_calls += 1
        assert _text(target) == "financial-economics:wrds"
        assert kind == 1 and flags == 0
        if self.raise_read:
            raise RuntimeError(PASSWORD)
        if self.read_success:
            ctypes.cast(output, ctypes.POINTER(ctypes.POINTER(credentials._CREDENTIALW)))[0] = ctypes.pointer(self.saved)
        return self.read_success

    def last_error(self):
        return self.error

    def free(self, pointer):
        self.free_calls += 1
        assert ctypes.addressof(pointer.contents) == ctypes.addressof(self.saved)
        if self.raise_free:
            raise RuntimeError(PASSWORD)

    def prompt(self, info, target, context, auth_error, username, username_size,
               password, password_size, save, flags):
        self.prompt_calls += 1
        assert _text(target) == credentials.CREDENTIAL_TARGET
        ui = ctypes.cast(info, ctypes.POINTER(credentials._CREDUI_INFOW)).contents
        assert ui.cbSize == ctypes.sizeof(credentials._CREDUI_INFOW)
        assert not ui.hwndParent and not ui.hbmBanner
        assert "WRDS" in _text(ui.pszCaptionText)
        assert context is None and auth_error == 0
        assert username_size == 514 and password_size == 257
        assert flags == 0x40000 | 0x00080 | 0x00002
        assert ctypes.cast(save, ctypes.POINTER(credentials._BOOL))[0] == 0
        self.prompt_username, self.prompt_password = username, password
        for buffer, text in ((username, self.username), (password, self.password)):
            encoded = text.encode("utf-16-le") + b"\0\0"
            ctypes.memmove(buffer, encoded, len(encoded))
        if self.raise_prompt:
            raise RuntimeError(PASSWORD)
        return self.prompt_status

    def write(self, pointer, flags):
        self.write_calls += 1
        saved = ctypes.cast(pointer, ctypes.POINTER(credentials._CREDENTIALW)).contents
        assert flags == 0 and saved.Flags == 0 and saved.Type == 1
        assert _text(saved.TargetName) == credentials.CREDENTIAL_TARGET
        assert _text(saved.UserName) == self.username
        assert saved.Persist == 2  # current Windows user, persistent on this computer
        assert saved.AttributeCount == 0 and not saved.Attributes
        raw = ctypes.string_at(saved.CredentialBlob, saved.CredentialBlobSize)
        assert raw == self.password.encode("utf-16-le")  # no trailing terminator
        if self.raise_write:
            raise RuntimeError(PASSWORD)
        return self.write_success

    def assert_prompt_buffers_cleared(self):
        for buffer in (self.prompt_username, self.prompt_password):
            assert ctypes.string_at(buffer, ctypes.sizeof(buffer)) == bytes(ctypes.sizeof(buffer))


@pytest.fixture
def api(monkeypatch):
    native = FakeAPI()
    monkeypatch.setattr(credentials, "_load_api", lambda **kwargs: native)
    return native


def test_read_exact_target_preserves_utf16_and_frees_native_buffer(api, capsys):
    assert credentials.get_saved_login() == (USERNAME, PASSWORD)
    assert api.read_calls == 1 and api.free_calls == 1
    assert capsys.readouterr() == ("", "")


def test_absent_credential_returns_none_without_freeing_an_unallocated_buffer(api):
    api.read_success, api.error = False, 1168
    assert credentials.get_saved_login() is None
    assert api.read_calls == 1 and api.free_calls == 0


def test_non_windows_read_returns_none_without_loading_native_api(monkeypatch):
    monkeypatch.setattr(credentials.sys, "platform", "linux")
    assert credentials.get_saved_login() is None


def test_non_windows_setup_fails_without_prompting(monkeypatch):
    monkeypatch.setattr(credentials.sys, "platform", "linux")
    with pytest.raises(credentials.CredentialError, match="WINDOWS_REQUIRED"):
        credentials.setup_saved_login()


def test_api_load_error_is_sanitized(monkeypatch):
    monkeypatch.setattr(credentials.sys, "platform", "win32")
    def fail(*args, **kwargs):
        raise OSError(PASSWORD)
    monkeypatch.setattr(credentials.ctypes, "WinDLL", fail)
    with pytest.raises(credentials.CredentialError, match="API_UNAVAILABLE") as error:
        credentials.get_saved_login()
    assert PASSWORD not in "".join(traceback.format_exception(error.value))


def test_win32_bindings_use_unicode_pointer_signatures_without_calling_native_functions(monkeypatch):
    class Function:
        def __call__(self, *args):
            raise AssertionError("Binding tests must not call native credential functions")
    native = SimpleNamespace(CredReadW=Function(), CredWriteW=Function(), CredFree=Function())
    ui = SimpleNamespace(CredUIPromptForCredentialsW=Function())
    loaded = []
    def load(name, **kwargs):
        loaded.append(name)
        assert kwargs == {"use_last_error": True}
        return {"Advapi32.dll": native, "Credui.dll": ui}[name]
    monkeypatch.setattr(credentials.sys, "platform", "win32")
    monkeypatch.setattr(credentials.ctypes, "WinDLL", load)
    monkeypatch.setattr(credentials.ctypes, "get_last_error", lambda: 0, raising=False)
    api = credentials._load_api(prompt=True)
    assert loaded == ["Advapi32.dll", "Credui.dll"]
    assert api.read is native.CredReadW and api.write is native.CredWriteW
    assert api.free is native.CredFree and api.prompt is ui.CredUIPromptForCredentialsW
    assert api.read.argtypes == [credentials._PWCHAR, credentials._DWORD, credentials._DWORD,
                                ctypes.POINTER(ctypes.POINTER(credentials._CREDENTIALW))]
    assert api.write.argtypes == [ctypes.POINTER(credentials._CREDENTIALW), credentials._DWORD]
    assert api.free.argtypes == [ctypes.c_void_p] and api.free.restype is None
    assert api.prompt.argtypes == [ctypes.POINTER(credentials._CREDUI_INFOW), credentials._PWCHAR,
                                   ctypes.c_void_p, credentials._DWORD, credentials._PWCHAR,
                                   credentials._DWORD, credentials._PWCHAR, credentials._DWORD,
                                   ctypes.POINTER(credentials._BOOL), credentials._DWORD]
    assert api.read.restype is credentials._BOOL and api.prompt.restype is credentials._DWORD


@pytest.mark.parametrize("exception", [False, True])
def test_read_failure_is_sanitized_and_missing_is_the_only_silent_error(api, exception):
    api.read_success, api.error, api.raise_read = False, 5, exception
    with pytest.raises(credentials.CredentialError, match="READ_FAILED") as error:
        credentials.get_saved_login()
    assert PASSWORD not in "".join(traceback.format_exception(error.value))
    assert api.free_calls == 0


@pytest.mark.parametrize("change", ["odd_size", "empty_blob", "too_large", "missing_username", "wrong_type", "bad_utf16"])
def test_invalid_saved_credentials_fail_and_release_native_buffer(api, change):
    if change == "odd_size":
        api.saved.CredentialBlobSize = 1
    elif change == "empty_blob":
        api.saved.CredentialBlobSize = 0
    elif change == "too_large":
        api.saved.CredentialBlobSize = 2562
    elif change == "missing_username":
        api.saved.UserName = credentials._PWCHAR()
    elif change == "wrong_type":
        api.saved.Type = 2
    else:
        api.blob[0], api.blob[1] = 0, 0xD8  # lone high surrogate
        api.saved.CredentialBlobSize = 2
    with pytest.raises(credentials.CredentialError):
        credentials.get_saved_login()
    assert api.free_calls == 1


def test_free_errors_are_sanitized(api):
    api.raise_free = True
    with pytest.raises(credentials.CredentialError, match="FREE_FAILED") as error:
        credentials.get_saved_login()
    assert PASSWORD not in "".join(traceback.format_exception(error.value))


def test_setup_uses_native_ui_stores_utf16_locally_and_clears_buffers(api, capsys):
    assert credentials.setup_saved_login() is True
    assert api.read_calls == 0 and api.prompt_calls == api.write_calls == 1
    api.assert_prompt_buffers_cleared()
    assert capsys.readouterr() == ("", "")


def test_cancel_never_writes_and_still_clears_buffers(api, capsys):
    api.prompt_status = 1223
    assert credentials.setup_saved_login() is False
    assert api.write_calls == 0
    api.assert_prompt_buffers_cleared()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("failure, code", [("prompt_status", "PROMPT_FAILED"),
                                         ("write_status", "WRITE_FAILED"),
                                         ("prompt_exception", "SETUP_FAILED"),
                                         ("write_exception", "SETUP_FAILED")])
def test_setup_errors_are_sanitized_and_all_prompt_buffers_are_cleared(api, failure, code):
    if failure == "prompt_status":
        api.prompt_status = 87
    elif failure == "write_status":
        api.write_success = False
    elif failure == "prompt_exception":
        api.raise_prompt = True
    else:
        api.raise_write = True
    with pytest.raises(credentials.CredentialError, match=code) as error:
        credentials.setup_saved_login()
    assert PASSWORD not in "".join(traceback.format_exception(error.value))
    api.assert_prompt_buffers_cleared()


@pytest.mark.parametrize("field", ["username", "password"])
def test_empty_login_is_not_stored(api, field):
    setattr(api, field, "")
    with pytest.raises(credentials.CredentialError, match="INVALID"):
        credentials.setup_saved_login()
    assert api.write_calls == 0
    api.assert_prompt_buffers_cleared()


@pytest.mark.parametrize("outcome, expected, code", [(True, "stored", 0), (False, "cancelled", 2)])
def test_cli_prints_only_status(api, capsys, outcome, expected, code):
    api.prompt_status = 0 if outcome else 1223
    assert credentials.main([]) == code
    output = capsys.readouterr()
    assert expected in output.out and output.err == ""
    assert USERNAME not in output.out and PASSWORD not in output.out


def test_cli_errors_print_only_fixed_failure_status(api, capsys):
    api.raise_prompt = True
    assert credentials.main([]) == 1
    assert capsys.readouterr() == ("", "WRDS login setup failed.\n")


def test_native_structures_use_windows_field_sizes_on_all_test_platforms():
    assert ctypes.sizeof(credentials._WCHAR) == 2
    assert ctypes.sizeof(credentials._DWORD) == ctypes.sizeof(credentials._BOOL) == 4
    assert ctypes.sizeof(credentials._CREDENTIALW) == (80 if ctypes.sizeof(ctypes.c_void_p) == 8 else 52)
    assert ctypes.sizeof(credentials._CREDUI_INFOW) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 20)
