import json
import struct
import threading
import http.server
import os
import time
import sqlite3
from urllib.parse import urlparse
from typing import Dict, Any, Callable, get_type_hints, TypedDict, Optional, Annotated, List
from pathlib import Path

PLUGIN_NAME = 'pcm'
PLUGIN_HOTKEY = "Ctrl-Alt-M"

class JSONRPCError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data


class RPCRegistry:
    def __init__(self):
        self.methods: Dict[str, Callable] = {}

    def register(self, func: Callable) -> Callable:
        self.methods[func.__name__] = func
        return func

    def dispatch(self, method: str, params: Any) -> Any:
        if method not in self.methods:
            raise JSONRPCError(-32601, f"Method '{method}' not found")

        func = self.methods[method]
        hints = get_type_hints(func)

        hints.pop("return", None)

        if isinstance(params, list):
            if len(params) != len(hints):
                raise JSONRPCError(-32602, f"Invalid params: expected {len(hints)} arguments, got {len(params)}")
            converted_params = []
            for value, (param_name, expected_type) in zip(params, hints.items()):
                try:
                    if not isinstance(value, expected_type):
                        value = expected_type(value)
                    converted_params.append(value)
                except (ValueError, TypeError):
                    raise JSONRPCError(-32602, f"Invalid type for parameter '{param_name}': expected {expected_type.__name__}")
            return func(*converted_params)
        elif isinstance(params, dict):
            if set(params.keys()) != set(hints.keys()):
                raise JSONRPCError(-32602, f"Invalid params: expected {list(hints.keys())}")

            converted_params = {}
            for param_name, expected_type in hints.items():
                value = params.get(param_name)
                try:
                    if not isinstance(value, expected_type):
                        value = expected_type(value)
                    converted_params[param_name] = value
                except (ValueError, TypeError):
                    raise JSONRPCError(-32602, f"Invalid type for parameter '{param_name}': expected {expected_type.__name__}")

            return func(**converted_params)
        else:
            raise JSONRPCError(-32600, "Invalid Request: params must be array or object")


rpc_registry = RPCRegistry()


def jsonrpc(func: Callable) -> Callable:
    """Decorator to register a function as a JSON-RPC method"""
    global rpc_registry
    return rpc_registry.register(func)


class JSONRPCRequestHandler(http.server.BaseHTTPRequestHandler):
    def send_jsonrpc_error(self, code: int, message: str, id: Any = None):
        response = {"jsonrpc": "2.0", "error": {"code": code, "message": message}}
        if id is not None:
            response["id"] = id
        response_body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)

    def do_POST(self):
        global rpc_registry
        import traceback

        parsed_path = urlparse(self.path)
        if parsed_path.path != "/mcp":
            self.send_jsonrpc_error(-32098, "Invalid endpoint", None)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_jsonrpc_error(-32700, "Parse error: missing request body", None)
            return

        request_body = self.rfile.read(content_length)
        try:
            request = json.loads(request_body)
        except json.JSONDecodeError:
            self.send_jsonrpc_error(-32700, "Parse error: invalid JSON", None)
            return

        # Prepare the response
        response = {"jsonrpc": "2.0"}
        if request.get("id") is not None:
            response["id"] = request.get("id")

        try:
            # Basic JSON-RPC validation
            if not isinstance(request, dict):
                raise JSONRPCError(-32600, "Invalid Request")
            if request.get("jsonrpc") != "2.0":
                raise JSONRPCError(-32600, "Invalid JSON-RPC version")
            if "method" not in request:
                raise JSONRPCError(-32600, "Method not specified")

            # Dispatch the method
            result = rpc_registry.dispatch(request["method"], request.get("params", []))
            response["result"] = result

        except JSONRPCError as e:
            response["error"] = {"code": e.code, "message": e.message}
            if e.data is not None:
                response["error"]["data"] = e.data
        except IDAError as e:
            response["error"] = {
                "code": -32000,
                "message": e.message,
            }
        except Exception as e:
            traceback.print_exc()
            response["error"] = {
                "code": -32603,
                "message": "Internal error",
                "data": traceback.format_exc(),
            }

        try:
            response_body = json.dumps(response).encode("utf-8")
        except Exception as e:
            traceback.print_exc()
            response_body = json.dumps(
                {
                    "error": {
                        "code": -32603,
                        "message": "Internal error",
                        "data": traceback.format_exc(),
                    }
                }
            ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        # Suppress logging
        pass


class MCPHTTPServer(http.server.HTTPServer):
    allow_reuse_address = False


class Server:
    HOST = "localhost"
    PORT = 13337

    def __init__(self):
        self.server = None
        self.server_thread = None
        self.running = False

    def start(self):
        if self.running:
            print(f"[{PLUGIN_NAME}] Server is already running")
            return

        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.running = True
        self.server_thread.start()

    def stop(self):
        if not self.running:
            return

        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join()
            self.server = None
        print(f"[{PLUGIN_NAME}] Server stopped")

    def _run_server(self):
        try:
            # Create server in the thread to handle binding
            self.server = MCPHTTPServer((Server.HOST, Server.PORT), JSONRPCRequestHandler)
            print(f"[{PLUGIN_NAME}] Server started at http://{Server.HOST}:{Server.PORT}")
            self.server.serve_forever()
        except OSError as e:
            if e.errno == 98 or e.errno == 10048:  # Port already in use (Linux/Windows)
                print(f"[{PLUGIN_NAME}] Error: Port 13337 is already in use")
            else:
                print(f"[{PLUGIN_NAME}] Server error: {e}")
            self.running = False
        except Exception as e:
            print(f"[{PLUGIN_NAME}] Server error: {e}")
        finally:
            self.running = False


# A module that helps with writing thread safe ida code.
# Based on:
# https://web.archive.org/web/20160305190440/http://www.williballenthin.com/blog/2015/09/04/idapython-synchronization-decorator/
import logging
import queue
import traceback
import functools

# import idapro
import ida_pro
import ida_hexrays
import ida_kernwin
import ida_funcs
import ida_entry
import ida_gdl
import ida_graph
import ida_lines
import ida_idaapi
import ida_name
import ida_segment
import ida_xref
import ida_typeinf
import idc
import idaapi
import idautils
import ida_nalt
import ida_bytes


class IDAError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    @property
    def message(self) -> str:
        return self.args[0]


class IDASyncError(Exception):
    pass


# Important note: Always make sure the return value from your function f is a
# copy of the data you have gotten from IDA, and not the original data.
#
# Example:
# --------
#
# Do this:
#
#   @idaread
#   def ts_Functions():
#       return list(idautils.Functions())
#
# Don't do this:
#
#   @idaread
#   def ts_Functions():
#       return idautils.Functions()
#

logger = logging.getLogger(__name__)


# Enum for safety modes. Higher means safer:
class IDASafety:
    ida_kernwin.MFF_READ
    SAFE_NONE = ida_kernwin.MFF_FAST
    SAFE_READ = ida_kernwin.MFF_READ
    SAFE_WRITE = ida_kernwin.MFF_WRITE


call_stack = queue.LifoQueue()


def sync_wrapper(ff, safety_mode: IDASafety):
    """
    Call a function ff with a specific IDA safety_mode.
    """
    # logger.debug('sync_wrapper: {}, {}'.format(ff.__name__, safety_mode))

    if safety_mode not in [IDASafety.SAFE_READ, IDASafety.SAFE_WRITE]:
        error_str = 'Invalid safety mode {} over function {}'.format(safety_mode, ff.__name__)
        logger.error(error_str)
        raise IDASyncError(error_str)

    # No safety level is set up:
    res_container = queue.Queue()

    def runned():
        # logger.debug('Inside runned')

        # Make sure that we are not already inside a sync_wrapper:
        if not call_stack.empty():
            last_func_name = call_stack.get()
            error_str = ('Call stack is not empty while calling the ' 'function {} from {}').format(ff.__name__, last_func_name)
            # logger.error(error_str)
            raise IDASyncError(error_str)

        call_stack.put((ff.__name__))
        try:
            res_container.put(ff())
        except Exception as x:
            res_container.put(x)
        finally:
            call_stack.get()
            # logger.debug('Finished runned')

    ret_val = idaapi.execute_sync(runned, safety_mode)
    res = res_container.get()
    if isinstance(res, Exception):
        raise res
    return res


def idawrite(f):
    """
    decorator for marking a function as modifying the IDB.
    schedules a request to be made in the main IDA loop to avoid IDB corruption.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ff = functools.partial(f, *args, **kwargs)
        ff.__name__ = f.__name__
        return sync_wrapper(ff, idaapi.MFF_WRITE)

    return wrapper


def idaread(f):
    """
    decorator for marking a function as reading from the IDB.
    schedules a request to be made in the main IDA loop to avoid
      inconsistent results.
    MFF_READ constant via: http://www.openrce.org/forums/posts/1827
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ff = functools.partial(f, *args, **kwargs)
        ff.__name__ = f.__name__
        return sync_wrapper(ff, idaapi.MFF_READ)

    return wrapper


def init_notes_db():
    user_dir = Path.home()
    if os.name == 'nt':
        db_path = user_dir / "AppData" / "Local" / "IDA_MCP"
    else:
        db_path = user_dir / ".ida_mcp"

    db_path.mkdir(exist_ok=True)
    db_file = db_path / "analysis_notes.db"

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    cursor.execute(
        '''
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY,
        file_md5 TEXT NOT NULL,
        address TEXT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        tags TEXT
    )
    '''
    )

    cursor.execute(
        '''
    CREATE TABLE IF NOT EXISTS files (
        md5 TEXT PRIMARY KEY,
        path TEXT NOT NULL,
        name TEXT NOT NULL,
        base_addr TEXT,
        size TEXT,
        sha256 TEXT,
        crc32 TEXT,
        filesize TEXT,
        last_accessed INTEGER
    )
    '''
    )

    conn.commit()
    conn.close()

    return str(db_file)


NOTES_DB = init_notes_db()


# Type definitions
class Function(TypedDict):
    start_address: int
    end_address: int
    name: str
    prototype: str


class Entrypoint(TypedDict):
    address: int
    name: str
    ordinal: int


class Block(TypedDict):
    start_address: int
    end_address: int
    type: str
    successor_addresses: List[int]


class CFGNode(TypedDict):
    id: int
    start_address: int
    end_address: int
    type: str
    successors: List[int]


class XrefEntry(TypedDict):
    from_address: int
    to_address: int
    type: str
    function_name: str


class Type(TypedDict):
    name: str
    definition: str
    size: int


class Note(TypedDict):
    id: int
    file_md5: str
    address: Optional[str]
    title: str
    content: str
    timestamp: int
    tags: Optional[str]


class FileInfo(TypedDict):
    md5: str
    path: str
    name: str
    base_addr: str
    size: str
    sha256: str
    crc32: str
    filesize: str
    last_accessed: int


class Metadata(TypedDict):
    path: str
    module: str
    base: str
    size: str
    md5: str
    sha256: str
    crc32: str
    filesize: str


def get_function(address: int) -> Optional[Function]:
    fn = idaapi.get_func(address)
    if fn is None:
        raise IDAError(f"No function found at address {address}")
    # NOTE: You need IDA 9.0 SP1 or newer for this
    prototype: ida_typeinf.tinfo_t = fn.get_prototype()
    if prototype is not None:
        prototype = str(prototype)
    return {
        "start_address": fn.start_ea,
        "end_address": fn.end_ea,
        "name": fn.name,
        "prototype": prototype,
    }


def get_image_size():
    import ida_ida

    omin_ea = ida_ida.inf_get_omin_ea()
    omax_ea = ida_ida.inf_get_omax_ea()
    # Bad heuristic for image size (bad if the relocations are the last section)
    image_size = omax_ea - omin_ea
    # Try to extract it from the PE header
    header = idautils.peutils_t().header()
    if header and header[:4] == b"PE\0\0":
        image_size = struct.unpack("<I", header[0x50:0x54])[0]
    return image_size


def decompile_checked(address: int) -> ida_hexrays.cfunc_t:
    if not ida_hexrays.init_hexrays_plugin():
        raise IDAError("Hex-Rays decompiler is not available")
    error = ida_hexrays.hexrays_failure_t()
    cfunc: ida_hexrays.cfunc_t = ida_hexrays.decompile_func(address, error, ida_hexrays.DECOMP_WARNINGS)
    if not cfunc:
        message = f"Decompilation failed at {address}"
        if error.str:
            message += f": {error.str}"
        if error.errea != idaapi.BADADDR:
            message += f" (address: {error.errea})"
        raise IDAError(message)
    return cfunc


def refresh_decompiler_widget():
    widget = ida_kernwin.get_current_widget()
    if widget is not None:
        vu = ida_hexrays.get_widget_vdui(widget)
        if vu is not None:
            vu.refresh_ctext()


def refresh_decompiler_ctext(function_address: int):
    error = ida_hexrays.hexrays_failure_t()
    cfunc: ida_hexrays.cfunc_t = ida_hexrays.decompile_func(function_address, error, ida_hexrays.DECOMP_WARNINGS)
    if cfunc:
        cfunc.refresh_func_ctext()


class my_modifier_t(ida_hexrays.user_lvar_modifier_t):
    def __init__(self, var_name: str, new_type: ida_typeinf.tinfo_t):
        ida_hexrays.user_lvar_modifier_t.__init__(self)
        self.var_name = var_name
        self.new_type = new_type

    def modify_lvars(self, lvars):
        for lvar_saved in lvars.lvvec:
            lvar_saved: ida_hexrays.lvar_saved_info_t
            if lvar_saved.name == self.var_name:
                lvar_saved.type = self.new_type
                return True
        return False


#
# Function and code analysis functions
#


@jsonrpc
@idaread
def get_function_by_name(name: Annotated[str, "Name of the function to get"]) -> Function:
    """Get a function by its name"""
    function_address = ida_name.get_name_ea(ida_idaapi.BADADDR, name)
    if function_address == ida_idaapi.BADADDR:
        raise IDAError(f"No function found with name {name}")
    return get_function(function_address)


@jsonrpc
@idaread
def get_function_by_address(address: Annotated[int, "Address of the function to get"]) -> Function:
    """Get a function by its address"""
    return get_function(address)


@jsonrpc
@idaread
def get_current_address() -> int:
    """Get the address currently selected by the user"""
    return idaapi.get_screen_ea()


@jsonrpc
@idaread
def get_current_function() -> Optional[Function]:
    """Get the function currently selected by the user"""
    return get_function(idaapi.get_screen_ea())


@jsonrpc
@idaread
def list_functions() -> list[Function]:
    """List all functions in the database"""
    return [get_function(address) for address in idautils.Functions()]


@jsonrpc
@idaread
def decompile_function(address: Annotated[int, "Address of the function to decompile"]) -> str:
    """Decompile a function at the given address using Hex-Rays"""
    cfunc = decompile_checked(address)
    sv = cfunc.get_pseudocode()
    cfunc.get_eamap()
    pseudocode = ""
    for i, sl in enumerate(sv):
        sl: ida_kernwin.simpleline_t
        item = ida_hexrays.ctree_item_t()
        addr = None if i > 0 else cfunc.entry_ea
        if cfunc.get_line_item(sl.line, 1, False, None, item, None):
            ds = item.dstr().split(": ")
            print(f"[{PLUGIN_NAME}] {ds = }")
            if len(ds) == 2:
                addr = int(ds[0], 16)
        line = ida_lines.tag_remove(sl.line)
        if len(pseudocode) > 0:
            pseudocode += "\n"
        if addr is None:
            pseudocode += f"/* line: {i} */ {line}"
        else:
            pseudocode += f"/* line: {i}, address: {addr} */ {line}"

    return pseudocode


@jsonrpc
@idaread
def disassemble_function(address: Annotated[int, "Address of the function to disassemble"]) -> str:
    """Get assembly code (address: instruction; comment) for a function"""
    func = idaapi.get_func(address)
    if not func:
        raise IDAError(f"No function found at address {address}")

    # TODO: add labels
    disassembly = ""
    for address in ida_funcs.func_item_iterator_t(func):
        if len(disassembly) > 0:
            disassembly += "\n"
        disassembly += f"{address}: "
        disassembly += idaapi.generate_disasm_line(address, idaapi.GENDSM_REMOVE_TAGS)
        comment = idaapi.get_cmt(address, False)
        if not comment:
            comment = idaapi.get_cmt(address, True)
        if comment:
            disassembly += f"; {comment}"
    return disassembly


@jsonrpc
@idaread
def get_entrypoints() -> List[Entrypoint]:
    """Get all entrypoints in the binary"""
    entrypoints = []

    for i in range(ida_entry.get_entry_qty()):
        ordinal = i
        address = ida_entry.get_entry(ordinal)
        name = ida_name.get_name(address)

        entrypoints.append({"address": address, "name": name if name else f"entry_{ordinal}", "ordinal": ordinal})

    return entrypoints


@jsonrpc
@idaread
def get_function_blocks(address: Annotated[int, "Address of the function to get blocks for"]) -> List[Block]:
    """Get all basic blocks in a function"""
    func = idaapi.get_func(address)
    if not func:
        raise IDAError(f"No function found at address {address}")

    # Get control flow graph
    flow_chart = ida_gdl.FlowChart(func)
    blocks = []

    for block in flow_chart:
        successor_addresses = []
        for succ_idx in range(block.nsucc()):
            succ_block = block.succ(succ_idx)
            successor_addresses.append(succ_block.start_ea)

        blocks.append({"start_address": block.start_ea, "end_address": block.end_ea, "type": "block", "successor_addresses": successor_addresses})  # Default block type

    return blocks


@jsonrpc
@idaread
def get_function_cfg(address: Annotated[int, "Address of the function to get CFG for"]) -> List[CFGNode]:
    """Get control flow graph for a function"""
    func = idaapi.get_func(address)
    if not func:
        raise IDAError(f"No function found at address {address}")

    # Get control flow graph
    flow_chart = ida_gdl.FlowChart(func)
    nodes = []

    for i, block in enumerate(flow_chart):
        successors = []
        for succ_idx in range(block.nsucc()):
            succ_block = block.succ(succ_idx)
            # Store the block ID as successor
            successors.append(succ_block.id)

        # Determine block type
        block_type = "normal"
        if i == 0:
            block_type = "entry"
        elif block.nsucc() == 0:
            block_type = "exit"

        nodes.append({"id": block.id, "start_address": block.start_ea, "end_address": block.end_ea, "type": block_type, "successors": successors})

    return nodes


@jsonrpc
@idaread
def get_xrefs_to(address: Annotated[int, "Address to get xrefs to"]) -> List[XrefEntry]:
    """Get all cross references to the given address"""
    xrefs = []

    xref = ida_xref.get_first_cref_to(address)
    while xref != ida_idaapi.BADADDR:
        xref_type = "unknown"
        if ida_xref.is_call_insn(xref):
            xref_type = "call"
        elif ida_xref.is_flow(xref):
            xref_type = "flow"
        else:
            xref_type = "data"

        # Get the function containing this xref
        func = idaapi.get_func(xref)
        function_name = ida_funcs.get_func_name(func.start_ea) if func else "global"

        xrefs.append({"from_address": xref, "to_address": address, "type": xref_type, "function_name": function_name})

        xref = ida_xref.get_next_cref_to(address, xref)

    return xrefs


@jsonrpc
@idaread
def get_xrefs_from(address: Annotated[int, "Address to get xrefs from"]) -> List[XrefEntry]:
    """Get all cross references from the given address"""
    xrefs = []

    xref = ida_xref.get_first_cref_from(address)
    while xref != ida_idaapi.BADADDR:
        xref_type = "unknown"
        if ida_xref.is_call_insn(address):
            xref_type = "call"
        elif ida_xref.is_flow(xref):
            xref_type = "flow"
        else:
            xref_type = "data"

        # Get the function containing this xref
        func = idaapi.get_func(address)
        function_name = ida_funcs.get_func_name(func.start_ea) if func else "global"

        xrefs.append({"from_address": address, "to_address": xref, "type": xref_type, "function_name": function_name})

        xref = ida_xref.get_next_cref_from(address, xref)

    return xrefs


#
# Modification functions
#


@jsonrpc
@idawrite
def set_decompiler_comment(address: Annotated[int, "Address in the function to set the comment for"], comment: Annotated[str, "Comment text (not shown in the disassembly)"]):
    """Set a comment for a given address in the function pseudocode"""

    # Reference: https://cyber.wtf/2019/03/22/using-ida-python-to-analyze-trickbot/
    # Check if the address corresponds to a line
    cfunc = decompile_checked(address)

    # Special case for function entry comments
    if address == cfunc.entry_ea:
        idc.set_func_cmt(address, comment, True)
        cfunc.refresh_func_ctext()
        return

    eamap = cfunc.get_eamap()
    if address not in eamap:
        raise IDAError(f"Failed to set comment at {address}")
    nearest_ea = eamap[address][0].ea

    # Remove existing orphan comments
    if cfunc.has_orphan_cmts():
        cfunc.del_orphan_cmts()
        cfunc.save_user_cmts()

    # Set the comment by trying all possible item types
    tl = idaapi.treeloc_t()
    tl.ea = nearest_ea
    for itp in range(idaapi.ITP_SEMI, idaapi.ITP_COLON):
        tl.itp = itp
        cfunc.set_user_cmt(tl, comment)
        cfunc.save_user_cmts()
        cfunc.refresh_func_ctext()
        if not cfunc.has_orphan_cmts():
            return
        cfunc.del_orphan_cmts()
        cfunc.save_user_cmts()
    raise IDAError(f"Failed to set comment at {address}")


@jsonrpc
@idawrite
def set_disassembly_comment(address: Annotated[int, "Address in the function to set the comment for"], comment: Annotated[str, "Comment text (not shown in the pseudocode)"]):
    """Set a comment for a given address in the function disassembly"""
    if not idaapi.set_cmt(address, comment, False):
        raise IDAError(f"Failed to set comment at {address}")


@jsonrpc
@idawrite
def rename_local_variable(
    function_address: Annotated[int, "Address of the function containing the variable"], old_name: Annotated[str, "Current name of the variable"], new_name: Annotated[str, "New name for the variable"]
):
    """Rename a local variable in a function"""
    func = idaapi.get_func(function_address)
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    if not ida_hexrays.rename_lvar(func.start_ea, old_name, new_name):
        raise IDAError(f"Failed to rename local variable {old_name} in function at {func.start_ea}")
    refresh_decompiler_ctext(func.start_ea)
    return True


@jsonrpc
@idawrite
def rename_function(function_address: Annotated[int, "Address of the function to rename"], new_name: Annotated[str, "New name for the function"]):
    """Rename a function"""
    fn = idaapi.get_func(function_address)
    if not fn:
        raise IDAError(f"No function found at address {function_address}")
    result = idaapi.set_name(fn.start_ea, new_name)
    refresh_decompiler_ctext(fn.start_ea)
    return result


@jsonrpc
@idawrite
def set_function_prototype(function_address: Annotated[int, "Address of the function"], prototype: Annotated[str, "New function prototype"]) -> bool:
    """Set a function's prototype"""
    fn = idaapi.get_func(function_address)
    if not fn:
        raise IDAError(f"No function found at address {function_address}")
    try:
        tif = ida_typeinf.tinfo_t()
        if not tif.get_named_type(ida_typeinf.get_idati(), prototype):
            if not tif.create_func(prototype):
                raise IDAError(f"Failed to parse prototype string: {prototype}")
        if not ida_typeinf.apply_tinfo(fn.start_ea, tif, ida_typeinf.TINFO_DEFINITE):
            raise IDAError(f"Failed to apply type")
        refresh_decompiler_ctext(fn.start_ea)
        return True
    except Exception as e:
        raise IDAError(f"Failed to parse prototype string: {prototype}. Error: {str(e)}")


@jsonrpc
@idawrite
def set_local_variable_type(
    function_address: Annotated[int, "Address of the function containing the variable"], variable_name: Annotated[str, "Name of the variable"], new_type: Annotated[str, "New type for the variable"]
) -> bool:
    """Set a local variable's type"""
    try:
        new_tif = ida_typeinf.tinfo_t()
        if not new_tif.get_named_type(ida_typeinf.get_idati(), new_type):
            raise IDAError(f"Failed to parse type: {new_type}")
    except Exception as e:
        raise IDAError(f"Failed to parse type: {new_type}. Error: {str(e)}")

    fn = idaapi.get_func(function_address)
    if not fn:
        raise IDAError(f"No function found at address {function_address}")
    if not ida_hexrays.rename_lvar(fn.start_ea, variable_name, variable_name):
        raise IDAError(f"Failed to find local variable: {variable_name}")

    try:
        modifier = my_modifier_t(variable_name, new_tif)
        if not ida_hexrays.modify_user_lvars(fn.start_ea, modifier):
            raise IDAError(f"Failed to modify local variable: {variable_name}")
        refresh_decompiler_ctext(fn.start_ea)
        return True
    except Exception as e:
        raise IDAError(f"Failed to modify local variable: {variable_name}. Error: {str(e)}")


@jsonrpc
@idawrite
def create_structure_type(
    name: Annotated[str, "Name of the new structure"],
    members: Annotated[List[Dict[str, str]], "List of structure members with name and type"],
    is_union: Annotated[bool, "Whether this is a union (True) or struct (False)"] = False,
) -> bool:
    """Create a new structure type"""
    try:
        # Check if structure with this name already exists
        existing_id = idc.get_struc_id(name)
        if existing_id != ida_idaapi.BADADDR:
            idc.del_struc(idc.get_struc(existing_id))

        # Create new structure
        sid = idc.add_struc(ida_idaapi.BADADDR, name, is_union)
        if sid == ida_idaapi.BADADDR:
            raise IDAError(f"Failed to create structure {name}")

        sptr = idc.get_struc(sid)
        if not sptr:
            raise IDAError(f"Failed to get structure pointer for {name}")

        # Add members to structure
        for member in members:
            member_name = member.get("name", "")
            member_type = member.get("type", "")
            member_offset = -1  # Let IDA choose the next offset

            tif = ida_typeinf.tinfo_t()
            if not tif.get_named_type(ida_typeinf.get_idati(), member_type):
                # Try to create a basic type
                if not ida_typeinf.parse_decl(tif, ida_typeinf.get_idati(), f"{member_type};", ida_typeinf.PT_SIL):
                    raise IDAError(f"Failed to parse type {member_type} for member {member_name}")

            # Add member
            if idc.add_struc_member(sptr, member_name, member_offset, ida_bytes.byteflag(), None, ida_typeinf.get_type_size(ida_typeinf.get_idati(), tif)) != 0:
                raise IDAError(f"Failed to add member {member_name} to structure {name}")

            # Set member type
            member_idx = idc.get_member_by_name(sptr, member_name)
            if member_idx is None:
                raise IDAError(f"Failed to get member index for {member_name}")

            member_ptr = idc.get_member(sptr, member_idx)
            if member_ptr is None:
                raise IDAError(f"Failed to get member pointer for {member_name}")

            if not ida_typeinf.set_member_tinfo(ida_typeinf.get_idati(), sptr, member_ptr, 0, tif, ida_typeinf.SET_MEMTI_COMPATIBLE):
                raise IDAError(f"Failed to set type for member {member_name}")

        return True
    except Exception as e:
        raise IDAError(f"Failed to create structure {name}. Error: {str(e)}")


@jsonrpc
@idaread
def get_metadata() -> Metadata:
    """Get metadata about the current IDB"""
    return {
        "path": idaapi.get_input_file_path(),
        "module": idaapi.get_root_filename(),
        "base": hex(idaapi.get_imagebase()),
        "size": hex(get_image_size()),
        "md5": ida_nalt.retrieve_input_file_md5().hex(),
        "sha256": ida_nalt.retrieve_input_file_sha256().hex(),
        "crc32": hex(ida_nalt.retrieve_input_file_crc32()),
        "filesize": hex(ida_nalt.retrieve_input_file_size()),
    }


#
# Notes and multi-binary support functions
#


@jsonrpc
def add_note(
    title: Annotated[str, "Title of the note"],
    content: Annotated[str, "Content of the note"],
    address: Annotated[Optional[int], "Address this note is related to (optional)"] = None,
    tags: Annotated[Optional[str], "Comma-separated tags for this note"] = None,
) -> int:
    """Add a new analysis note for the current binary"""

    # Get current file metadata
    metadata = get_metadata()
    file_md5 = metadata["md5"]

    # Store file info if not already present
    conn = sqlite3.connect(NOTES_DB)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM files WHERE md5 = ?", (file_md5,))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO files (md5, path, name, base_addr, size, sha256, crc32, filesize, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (file_md5, metadata["path"], metadata["module"], metadata["base"], metadata["size"], metadata["sha256"], metadata["crc32"], metadata["filesize"], int(time.time())),
        )
    else:
        # Update last accessed time
        cursor.execute("UPDATE files SET last_accessed = ? WHERE md5 = ?", (int(time.time()), file_md5))

    # Add note
    timestamp = int(time.time())
    address_str = hex(address) if address is not None else None

    cursor.execute("INSERT INTO notes (file_md5, address, title, content, timestamp, tags) VALUES (?, ?, ?, ?, ?, ?)", (file_md5, address_str, title, content, timestamp, tags))

    note_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return note_id


@jsonrpc
def update_note(
    note_id: Annotated[int, "ID of the note to update"],
    title: Annotated[Optional[str], "New title (or None to keep current)"] = None,
    content: Annotated[Optional[str], "New content (or None to keep current)"] = None,
    tags: Annotated[Optional[str], "New tags (or None to keep current)"] = None,
) -> bool:
    """Update an existing note"""

    conn = sqlite3.connect(NOTES_DB)
    cursor = conn.cursor()

    # Get current note
    cursor.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
    note = cursor.fetchone()
    if not note:
        conn.close()
        raise IDAError(f"Note with ID {note_id} not found")

    # Build update query
    update_parts = []
    params = []

    if title is not None:
        update_parts.append("title = ?")
        params.append(title)

    if content is not None:
        update_parts.append("content = ?")
        params.append(content)

    if tags is not None:
        update_parts.append("tags = ?")
        params.append(tags)

    if not update_parts:
        conn.close()
        return False  # Nothing to update

    # Update timestamp
    update_parts.append("timestamp = ?")
    params.append(int(time.time()))

    # Execute update
    params.append(note_id)
    cursor.execute(f"UPDATE notes SET {', '.join(update_parts)} WHERE id = ?", params)

    conn.commit()
    conn.close()

    return True


@jsonrpc
def get_notes(
    file_md5: Annotated[Optional[str], "MD5 of file to get notes for (or None for current file)"] = None,
    address: Annotated[Optional[int], "Get notes for specific address (optional)"] = None,
    tag: Annotated[Optional[str], "Filter notes by tag (optional)"] = None,
) -> List[Note]:
    """Get analysis notes for a binary"""

    # If no file_md5 specified, use current file
    if file_md5 is None:
        metadata = get_metadata()
        file_md5 = metadata["md5"]

    conn = sqlite3.connect(NOTES_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM notes WHERE file_md5 = ?"
    params = [file_md5]

    if address is not None:
        query += " AND address = ?"
        params.append(hex(address))

    if tag is not None:
        # Search for tag in comma-separated list
        query += " AND tags LIKE ?"
        params.append(f"%{tag}%")

    query += " ORDER BY timestamp DESC"

    cursor.execute(query, params)
    notes = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return notes


@jsonrpc
def delete_note(note_id: Annotated[int, "ID of the note to delete"]) -> bool:
    """Delete an analysis note"""

    conn = sqlite3.connect(NOTES_DB)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return deleted


@jsonrpc
def list_analyzed_files() -> List[FileInfo]:
    """List all previously analyzed files"""

    conn = sqlite3.connect(NOTES_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM files ORDER BY last_accessed DESC")
    files = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return files



class MCP(idaapi.plugin_t):
    flags = idaapi.PLUGIN_KEEP
    comment = "Model Context Protocol Plugin"
    help = "Enables MCP integration for remotely controlling IDA Pro"
    wanted_name = PLUGIN_NAME
    wanted_hotkey = PLUGIN_HOTKEY

    def init(self):
        self.server = Server()
        print(f"[{PLUGIN_NAME}] Plugin loaded, use Edit -> Plugins -> {PLUGIN_NAME} ({PLUGIN_HOTKEY}) to start the server")
        return idaapi.PLUGIN_KEEP

    def run(self, args):
        self.server.start()

    def term(self):
        self.server.stop()


def PLUGIN_ENTRY():
    return MCP()
