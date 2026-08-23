import argparse
import socket
import socketserver
import threading
import time
import webbrowser
import wsgiref.simple_server
import wsgiref.util
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
TOKEN_PATH = BASE_DIR / "token.json"


class _DualStackWSGIServer(socketserver.ThreadingMixIn, wsgiref.simple_server.WSGIServer):
    """Threaded WSGI-Server, der sowohl IPv4 (127.0.0.1) als auch IPv6 (::1) bedient.

    Wird gebraucht, weil Browser 'localhost' gemaess /etc/hosts oft zuerst
    als ::1 aufloesen; ein rein-IPv4-Server wuerde dann ERR_CONNECTION_REFUSED
    liefern. Threaded, damit Preconnect-/favicon-Requests den Callback nicht
    blockieren.
    """

    daemon_threads = True
    address_family = socket.AF_INET6

    def server_bind(self):
        if self.address_family == socket.AF_INET6:
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except OSError:
                pass
        super().server_bind()


class _RedirectWSGIApp:
    """Nimmt den OAuth-Callback entgegen und merkt sich die Request-URI."""

    def __init__(self, success_message):
        self.last_request_uri = None
        self._success_message = success_message

    def __call__(self, environ, start_response):
        start_response("200 OK", [("Content-type", "text/plain; charset=utf-8")])
        self.last_request_uri = wsgiref.util.request_uri(environ)
        return [self._success_message.encode("utf-8")]


def run_local_server_dualstack(flow, timeout_seconds=600):
    """Einmaliger OAuth-Login mit lokalem Callback-Server (dual-stack, threaded).

    Ersetzt InstalledAppFlow.run_local_server, das nur eine Request behandelt
    und nur eine IP-Familie bindet, was im Browser ERR_CONNECTION_REFUSED oder
    einen Timeout verursachen kann.
    """
    app = _RedirectWSGIApp(
        "The authentication flow has completed. You may close this window."
    )
    wsgiref.simple_server.WSGIServer.allow_reuse_address = False
    try:
        server = wsgiref.simple_server.make_server(
            "::", 0, app, server_class=_DualStackWSGIServer
        )
    except OSError:

        class _IPv4Only(_DualStackWSGIServer):
            address_family = socket.AF_INET

        server = wsgiref.simple_server.make_server(
            "0.0.0.0", 0, app, server_class=_IPv4Only
        )

    flow.redirect_uri = f"http://localhost:{server.server_port}/"
    auth_url, _ = flow.authorization_url()
    print(f"Bitte oeffne diese URL und melde dich an:\n{auth_url}")
    webbrowser.open(auth_url, new=1, autoraise=True)

    # Warten, bis der echte Callback mit 'code=' eintrifft. Andere Requests
    # (favicon, Preconnect) werden in eigenen Threads bedient und ignoriert.
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if app.last_request_uri is not None and "code=" in app.last_request_uri:
            break
        time.sleep(0.1)
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)

    if app.last_request_uri is None or "code=" not in app.last_request_uri:
        raise TimeoutError(
            "Timed out waiting for response from authorization server"
        )

    authorization_response = app.last_request_uri.replace("http", "https")
    flow.fetch_token(authorization_response=authorization_response)
    return flow.credentials


def get_credentials():
    """Liefert gespeicherte Credentials oder startet einmaligen OAuth-Login.

    Wenn token.json existiert, wird es geladen und bei Bedarf refresht.
    Sonst oeffnet sich der Browser (einmalige Anmeldung) und das Token
    wird dauerhaft in token.json gespeichert.
    """
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("Öffne Browser für einmalige Google-Anmeldung ...")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH), SCOPES
            )
            creds = run_local_server_dualstack(flow, timeout_seconds=600)
        TOKEN_PATH.write_text(creds.to_json())
    return creds


def upload_basic(file_path: str, file_name: str, mimetype: str = "video/mp4"):
    """Insert new file.
    Returns : Id's of the file uploaded

    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """
    creds = get_credentials()

    try:
        # create drive api client
        service = build("drive", "v3", credentials=creds)

        file_metadata = {"name": f"{file_name}"}
        media = MediaFileUpload(f"{file_path}", mimetype=mimetype)
        # pylint: disable=maybe-no-member
        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        print(f'File ID: {file.get("id")}')

    except HttpError as error:
        print(f"An error occurred: {error}")
        file = None

    return file.get("id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload eine Datei zu Google Drive")
    parser.add_argument("--file", type=str, required=True, help="Pfad zur hochzuladenden Datei")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name der Datei in Google Drive (Default: Dateiname)",
    )
    parser.add_argument(
        "--mimetype",
        type=str,
        default="video/mp4",
        help="MIME-Typ der Datei",
    )
    args = parser.parse_args()

    upload_basic(
        file_path=args.file,
        file_name=args.name if args.name else Path(args.file).name,
        mimetype=args.mimetype,
    )