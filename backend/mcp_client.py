"""The web app's MCP client: how the Gemini agent reaches the tools.

At startup the API launches backend/mcp_server.py as a child process and keeps
one MCP session open to it over stdio. The agent then:
  1. lists the server's tools (names, descriptions, JSON Schemas) to give Gemini,
  2. executes each Gemini function call as an MCP call_tool,
  3. reads plot figures from the server's figure:// resources for the UI.

If the child process cannot start, the same calls go to the same MCPServer
object in-process: identical tool registry and handlers, minus the stdio hop,
so the app keeps working and there is still only one tool definition.
"""

import asyncio
import concurrent.futures
import json
import logging
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver.exceptions import ToolError

from backend.config import BASE_DIR

logger = logging.getLogger(__name__)

START_TIMEOUT_S = 60.0
CALL_TIMEOUT_S = 90.0


class StdioConnection:
    """One long-lived stdio session to backend/mcp_server.py.

    The session lives on a private event loop in a background thread, so the
    synchronous agent code (running in FastAPI's threadpool) can submit MCP
    requests to it with run_coroutine_threadsafe.
    """

    def __init__(self) -> None:
        # Proactor is the Windows loop type that can spawn subprocesses.
        self._loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True)
        self._session: Optional[ClientSession] = None
        self._stop: Optional[asyncio.Event] = None
        self._runner: Optional[concurrent.futures.Future] = None
        self.tools: List[Any] = []

    async def _main(self, ready: concurrent.futures.Future) -> None:
        params = StdioServerParameters(command=sys.executable, args=["-m", "backend.mcp_server"],
                                       cwd=str(BASE_DIR))
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.tools = (await session.list_tools()).tools
                    self._session = session
                    self._stop = asyncio.Event()
                    ready.set_result(True)
                    await self._stop.wait()
        except BaseException as e:
            if not ready.done():
                ready.set_exception(e)
            raise
        finally:
            self._session = None

    def start(self) -> None:
        self._thread.start()
        ready: concurrent.futures.Future = concurrent.futures.Future()
        self._runner = asyncio.run_coroutine_threadsafe(self._main(ready), self._loop)
        try:
            ready.result(timeout=START_TIMEOUT_S)
        except BaseException:
            self._shutdown_loop()
            raise

    def _submit(self, coro: Any) -> Any:
        if self._session is None:
            raise RuntimeError("MCP session is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=CALL_TIMEOUT_S)

    def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        return self._submit(self._session.call_tool(name, args))

    def read_resource(self, uri: str) -> Any:
        return self._submit(self._session.read_resource(uri))

    def stop(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._runner is not None:
            try:
                self._runner.result(timeout=10)
            except Exception:
                pass
        self._shutdown_loop()

    def _shutdown_loop(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=5)


_conn: Optional[StdioConnection] = None
_lock = threading.Lock()


def start() -> str:
    """Launch the MCP server over stdio; fall back to in-process if it fails."""
    global _conn
    with _lock:
        if _conn is not None:
            return "stdio"
        conn = StdioConnection()
        try:
            conn.start()
        except Exception as e:  # missing interpreter, import error in the server, timeout...
            logger.warning("MCP server over stdio unavailable (%s); using it in-process", e)
            return "in-process"
        _conn = conn
        logger.info("MCP server running over stdio with %d tools", len(conn.tools))
        return "stdio"


def stop() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.stop()
            _conn = None


def transport() -> str:
    return "stdio" if _conn is not None else "in-process"


def _local_server() -> Any:
    from backend.mcp_server import mcp  # imported lazily: only needed for the fallback
    return mcp


def list_tools() -> List[Any]:
    """MCP tool definitions (name, description, input_schema)."""
    if _conn is not None:
        return list(_conn.tools)
    return asyncio.run(_local_server().list_tools())


def _error_text(text: str, name: str) -> str:
    # Drop the SDK's "Error executing tool <name>: " prefix; keep the reason.
    prefix = f"Error executing tool {name}: "
    return text[len(prefix):] if text.startswith(prefix) else text


def _read_figure(uri: str) -> Optional[str]:
    try:
        if _conn is not None:
            contents = _conn.read_resource(uri).contents
            return contents[0].text if contents else None
        contents = list(asyncio.run(_local_server().read_resource(uri)))
        return contents[0].content if contents else None
    except Exception as e:
        logger.warning("Could not read figure %s: %s", uri, e)
        return None


def call_tool(name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Run one tool through MCP. Returns (result_dict, figure_json_or_None);
    failures come back as {"error": reason} so the model can relay them."""
    try:
        if _conn is not None:
            result = _conn.call_tool(name, args)
        else:
            result = asyncio.run(_local_server().call_tool(name, args))
        text = result.content[0].text if result.content else ""
        if result.is_error:
            return {"error": _error_text(text, name)}, None
        data = json.loads(text)
    except Exception as e:
        # In-process the SDK raises instead of returning an error result.
        # ToolError = bad input the server reported on purpose; anything else is a bug.
        if not isinstance(e, ToolError):
            logger.exception("MCP tool %s failed", name)
        return {"error": _error_text(str(e), name)}, None

    if not isinstance(data, dict):
        return {"result": data}, None
    uri = data.pop("figure_uri", None)
    return data, (_read_figure(uri) if uri else None)
