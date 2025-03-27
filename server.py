import os
import sys
import ast
import json
import traceback
import shutil
import http.client
import signal
import asyncio
from typing import List, Optional, Dict, Any
from fastmcp import FastMCP

# The log_level is necessary for Cline to work: https://github.com/jlowin/fastmcp/issues/81
mcp = FastMCP("IDA Pro", log_level="ERROR")

jsonrpc_request_id = 1


def handle_shutdown(sig, frame):
    print("[*] Shutting down MCP server gracefully...")
    if hasattr(handle_shutdown, "_is_shutting_down") and handle_shutdown._is_shutting_down:
        return
    handle_shutdown._is_shutting_down = True

    if hasattr(asyncio, "get_running_loop"):
        try:
            loop = asyncio.get_running_loop()
            for task in asyncio.all_tasks(loop):
                task.cancel()
            print("[*] All tasks cancelled, shutdown complete.")
        except RuntimeError:
            pass
    import os
    os._exit(0)


def make_jsonrpc_request(method: str, *args):
    """Make a JSON-RPC request to the IDA plugin"""
    global jsonrpc_request_id
    conn = http.client.HTTPConnection("localhost", 13337)
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "params": list(args),
        "id": jsonrpc_request_id,
    }
    jsonrpc_request_id += 1

    try:
        conn.request("POST", "/mcp", json.dumps(request), {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = json.loads(response.read().decode())

        if "error" in data:
            error = data["error"]
            error_message = f"JSON-RPC error {error['code']}: {error['message']}"
            if "data" in error and error["data"]:
                error_message += f"\nDetails: {error['data']}"
            raise Exception(error_message)

        return data.get("result")
    except Exception as e:
        traceback.print_exc()
        raise
    finally:
        conn.close()


class MCPVisitor(ast.NodeVisitor):
    def __init__(self):
        self.types: dict[str, ast.ClassDef] = {}
        self.functions: dict[str, ast.FunctionDef] = {}
        self.descriptions: dict[str, str] = {}

    def visit_FunctionDef(self, node):
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id == "jsonrpc":
                    for i, arg in enumerate(node.args.args):
                        arg_name = arg.arg
                        arg_type = arg.annotation
                        if arg_type is None:
                            raise Exception(f"Missing argument type for {node.name}.{arg_name}")
                        if isinstance(arg_type, ast.Subscript):
                            assert isinstance(arg_type.value, ast.Name)
                            assert arg_type.value.id == "Annotated"
                            assert isinstance(arg_type.slice, ast.Tuple)
                            assert len(arg_type.slice.elts) == 2
                            annot_type = arg_type.slice.elts[0]
                            annot_description = arg_type.slice.elts[1]
                            assert isinstance(annot_description, ast.Constant)
                            node.args.args[i].annotation = ast.Subscript(
                                value=ast.Name(id="Annotated", ctx=ast.Load()),
                                slice=ast.Tuple(
                                    elts=[annot_type, ast.Call(func=ast.Name(id="Field", ctx=ast.Load()), args=[], keywords=[ast.keyword(arg="description", value=annot_description)])], ctx=ast.Load()
                                ),
                                ctx=ast.Load(),
                            )
                        elif isinstance(arg_type, ast.Name):
                            pass
                        else:
                            raise Exception(f"Unexpected type annotation for {node.name}.{arg_name} -> {type(arg_type)}")

                    body_comment = node.body[0]
                    if isinstance(body_comment, ast.Expr) and isinstance(body_comment.value, ast.Constant):
                        new_body = [body_comment]
                        self.descriptions[node.name] = body_comment.value.value
                    else:
                        new_body = []

                    call_args = [ast.Constant(value=node.name)]
                    for arg in node.args.args:
                        call_args.append(ast.Name(id=arg.arg, ctx=ast.Load()))
                    new_body.append(ast.Return(value=ast.Call(func=ast.Name(id="make_jsonrpc_request", ctx=ast.Load()), args=call_args, keywords=[])))
                    decorator_list = [ast.Call(func=ast.Attribute(value=ast.Name(id="mcp", ctx=ast.Load()), attr="tool", ctx=ast.Load()), args=[], keywords=[])]
                    node_nobody = ast.FunctionDef(node.name, node.args, new_body, decorator_list, node.returns, node.type_comment, lineno=node.lineno, col_offset=node.col_offset)
                    self.functions[node.name] = node_nobody

    def visit_ClassDef(self, node):
        for base in node.bases:
            if isinstance(base, ast.Name):
                if base.id == "TypedDict":
                    self.types[node.name] = node


SCRIPT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src')
IDA_PLUGIN_PY = os.path.join(SCRIPT_DIR, 'ida', "mcp.py")
GENERATED_PY = os.path.join(SCRIPT_DIR, "server_generated.py")

# NOTE: This is in the global scope on purpose
with open(IDA_PLUGIN_PY, "r") as f:
    code = f.read()
module = ast.parse(code, IDA_PLUGIN_PY)
visitor = MCPVisitor()
visitor.visit(module)
code = """# NOTE: This file has been automatically generated, do not modify!
from typing import Annotated, Optional, TypedDict, List
from pydantic import Field

"""
for type in visitor.types.values():
    code += ast.unparse(type)
    code += "\n\n"
for function in visitor.functions.values():
    code += ast.unparse(function)
    code += "\n\n"
with open(GENERATED_PY, "w") as f:
    f.write(code)
exec(compile(code, GENERATED_PY, "exec"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MCP server for IDA Pro")
    parser.add_argument("--generate-only", action="store_true", help="Generate the IDA plugin code and exit")
    parser.add_argument("--install-plugin", action="store_true", help="Install the IDA plugin")
    args = parser.parse_args()

    if args.generate_only:
        print(f"[*] Generating IDA plugin code...", file=sys.stderr)
        for function in visitor.functions.values():
            signature = function.name + "("
            for i, arg in enumerate(function.args.args):
                if i > 0:
                    signature += ", "
                signature += arg.arg
            signature += ")"
            description = visitor.descriptions.get(function.name, "<no description>")
            if description[-1] != ".":
                description += "."
            print(f"- `{signature}`: {description}")
        sys.exit(0)
    elif args.install_plugin:
        print(f"[*] Installing IDA plugin...", file=sys.stderr)
        if sys.platform == "win32":
            ida_plugin_folder = os.path.join(os.getenv("APPDATA"), "Hex-Rays", "IDA Pro", "plugins")
        else:
            ida_plugin_folder = os.path.join(os.path.expanduser("~"), ".idapro", "plugins")
        plugin_destination = os.path.join(ida_plugin_folder, "pcm.py")
        if input(f"Installing IDA plugin to {plugin_destination}, proceed? [Y/n] ").lower() == "n":
            sys.exit(1)
        if not os.path.exists(ida_plugin_folder):
            os.makedirs(ida_plugin_folder)
        shutil.copy(IDA_PLUGIN_PY, plugin_destination)
        print(f"Installed plugin: {plugin_destination}")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    print('[*] Starting MCP server...')
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        print("[*] Received keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"[!] Error: {e}")
        traceback.print_exc()
    finally:
        print("[*] MCP server stopped.")


if __name__ == "__main__":
    main()
