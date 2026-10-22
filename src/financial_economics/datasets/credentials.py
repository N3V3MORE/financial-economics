"""One WRDS login in Windows Credential Manager; no credential files or logging.

Run ``fecon wrds-credentials`` for native interactive
setup. Importing this module never loads credentials or displays a prompt.
"""

import argparse
import ctypes
from types import SimpleNamespace
import sys


CREDENTIAL_TARGET = "financial-economics:wrds"
_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2
_ERROR_NOT_FOUND = 1168
_ERROR_CANCELLED = 1223
_PROMPT_FLAGS = 0x40000 | 0x00080 | 0x00002  # generic, always show, do not persist
_MAX_USERNAME = 513
_MAX_PASSWORD = 256
_MAX_BLOB_BYTES = 5 * 512
_DWORD, _BOOL, _WCHAR = ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint16
_PWCHAR = ctypes.POINTER(_WCHAR)


class CredentialError(RuntimeError):
    """Fixed error codes keep native errors and credential values out of logs."""


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", _DWORD), ("dwHighDateTime", _DWORD)]


class _CREDENTIALW(ctypes.Structure):
    _fields_ = [("Flags", _DWORD), ("Type", _DWORD), ("TargetName", _PWCHAR),
                ("Comment", _PWCHAR), ("LastWritten", _FILETIME),
                ("CredentialBlobSize", _DWORD),
                ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
                ("Persist", _DWORD), ("AttributeCount", _DWORD),
                ("Attributes", ctypes.c_void_p), ("TargetAlias", _PWCHAR),
                ("UserName", _PWCHAR)]


class _CREDUI_INFOW(ctypes.Structure):
    _fields_ = [("cbSize", _DWORD), ("hwndParent", ctypes.c_void_p),
                ("pszMessageText", _PWCHAR), ("pszCaptionText", _PWCHAR),
                ("hbmBanner", ctypes.c_void_p)]


def _wide_buffer(text):
    encoded = text.encode("utf-16-le") + b"\0\0"
    return (_WCHAR * (len(encoded) // 2)).from_buffer_copy(encoded)


def _wide_length(pointer, capacity):
    for length in range(capacity):
        if pointer[length] == 0:
            return length
    raise CredentialError("WRDS_CREDENTIAL_INVALID")


def _load_api(*, prompt=False):
    if sys.platform != "win32":
        return None
    try:
        native = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
        native.CredReadW.argtypes = [_PWCHAR, _DWORD, _DWORD,
                                     ctypes.POINTER(ctypes.POINTER(_CREDENTIALW))]
        native.CredReadW.restype = _BOOL
        native.CredWriteW.argtypes = [ctypes.POINTER(_CREDENTIALW), _DWORD]
        native.CredWriteW.restype = _BOOL
        native.CredFree.argtypes = [ctypes.c_void_p]
        native.CredFree.restype = None
        api = SimpleNamespace(read=native.CredReadW, write=native.CredWriteW,
                              free=native.CredFree, last_error=ctypes.get_last_error)
        if prompt:
            ui = ctypes.WinDLL("Credui.dll", use_last_error=True)
            ui.CredUIPromptForCredentialsW.argtypes = [
                ctypes.POINTER(_CREDUI_INFOW), _PWCHAR, ctypes.c_void_p, _DWORD,
                _PWCHAR, _DWORD, _PWCHAR, _DWORD, ctypes.POINTER(_BOOL), _DWORD]
            ui.CredUIPromptForCredentialsW.restype = _DWORD
            api.prompt = ui.CredUIPromptForCredentialsW
        return api
    except Exception:
        raise CredentialError("WRDS_CREDENTIAL_API_UNAVAILABLE") from None


def get_saved_login():
    """Return this application's saved (username, password), or None if absent.

    Only the exact generic target is read. Native buffers are always released;
    the returned strings exist in memory only for the caller's database login.
    """
    api = _load_api()
    if api is None:
        return None
    target = _wide_buffer(CREDENTIAL_TARGET)
    credential = ctypes.POINTER(_CREDENTIALW)()
    try:
        try:
            if not api.read(target, _CRED_TYPE_GENERIC, 0, ctypes.byref(credential)):
                if api.last_error() == _ERROR_NOT_FOUND:
                    return None
                raise CredentialError("WRDS_CREDENTIAL_READ_FAILED")
            if not credential:
                raise CredentialError("WRDS_CREDENTIAL_INVALID")
            saved = credential.contents
            size = int(saved.CredentialBlobSize)
            if (saved.Type != _CRED_TYPE_GENERIC or not saved.UserName
                    or not saved.CredentialBlob or not 0 < size <= _MAX_BLOB_BYTES or size % 2):
                raise CredentialError("WRDS_CREDENTIAL_INVALID")
            length = _wide_length(saved.UserName, _MAX_USERNAME + 1)
            username = ctypes.string_at(saved.UserName, length * 2).decode("utf-16-le")
            password = ctypes.string_at(saved.CredentialBlob, size).decode("utf-16-le")
            if not username or not password or "\0" in password:
                raise CredentialError("WRDS_CREDENTIAL_INVALID")
            return username, password
        except CredentialError:
            raise
        except Exception:
            raise CredentialError("WRDS_CREDENTIAL_READ_FAILED") from None
    finally:
        if credential:
            try:
                api.free(credential)
            except Exception:
                raise CredentialError("WRDS_CREDENTIAL_FREE_FAILED") from None


def setup_saved_login():
    """Prompt natively and store the login for this user/computer; False on cancel."""
    api = _load_api(prompt=True)
    if api is None:
        raise CredentialError("WRDS_CREDENTIAL_WINDOWS_REQUIRED")
    target = _wide_buffer(CREDENTIAL_TARGET)
    message = _wide_buffer("Enter your WRDS database username (not your email address) "
                           "and WRDS password. They will be saved for this Windows user "
                           "on this computer in Windows Credential Manager.")
    caption = _wide_buffer("Financial economics: WRDS login")
    info = _CREDUI_INFOW(ctypes.sizeof(_CREDUI_INFOW), None, message, caption, None)
    username = (_WCHAR * (_MAX_USERNAME + 1))()
    password = (_WCHAR * (_MAX_PASSWORD + 1))()
    save = _BOOL(0)
    try:
        try:
            status = api.prompt(ctypes.byref(info), target, None, 0,
                                username, len(username), password, len(password),
                                ctypes.byref(save), _PROMPT_FLAGS)
            if status == _ERROR_CANCELLED:
                return False
            if status != 0:
                raise CredentialError("WRDS_CREDENTIAL_PROMPT_FAILED")
            if not username[0] or not password[0]:
                raise CredentialError("WRDS_CREDENTIAL_INVALID")
            _wide_length(username, len(username))
            size = _wide_length(password, len(password)) * 2
            credential = _CREDENTIALW()
            credential.Type = _CRED_TYPE_GENERIC
            credential.TargetName = target
            credential.UserName = username
            credential.CredentialBlobSize = size
            credential.CredentialBlob = ctypes.cast(password, ctypes.POINTER(ctypes.c_ubyte))
            credential.Persist = _CRED_PERSIST_LOCAL_MACHINE
            if not api.write(ctypes.byref(credential), 0):
                raise CredentialError("WRDS_CREDENTIAL_WRITE_FAILED")
            return True
        except CredentialError:
            raise
        except Exception:
            raise CredentialError("WRDS_CREDENTIAL_SETUP_FAILED") from None
    finally:
        ctypes.memset(username, 0, ctypes.sizeof(username))
        ctypes.memset(password, 0, ctypes.sizeof(password))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Save your WRDS login using the native Windows dialog.")
    parser.parse_args(argv)
    try:
        stored = setup_saved_login()
    except CredentialError:
        print("WRDS login setup failed.", file=sys.stderr)
        return 1
    print("WRDS login stored in Windows Credential Manager." if stored else "WRDS login setup cancelled.")
    return 0 if stored else 2


if __name__ == "__main__":
    raise SystemExit(main())
