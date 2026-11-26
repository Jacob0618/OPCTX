"""python main.py build ... / python main.py ask ..."""
import argparse
import getpass
import json
import os
from pathlib import Path
from config import PROFILES
from opctx import OpCtx

def main():
    parser = argparse.ArgumentParser(description="Minimal OpCtx")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build one independent history")
    build.add_argument("--dataset", choices=PROFILES, required=True)
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    ask = sub.add_parser("ask", help="Answer from a saved memory")
    ask.add_argument("--memory", type=Path, required=True)
    ask.add_argument("--question", required=True)
    args = parser.parse_args()
    if args.command == "build" and args.output.exists():
        raise FileExistsError("Output exists; choose a new path")
    if not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API key (not saved): ")
    if args.command == "build":
        payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
        sessions = payload["sessions"]
        if not isinstance(sessions, list):
            raise ValueError("Input must contain a sessions array")
        app = OpCtx(args.dataset)
        app.write(sessions, image_root=args.input.resolve().parent)
        # Exclusive create, so existing memories are never silently replaced.
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(app.memory, handle, ensure_ascii=False, allow_nan=False)
        print(f"Saved {len(app.memory['facts'])} facts and "
              f"{len(app.memory['views'])} event views to {args.output}")
    else:
        memory = json.loads(args.memory.read_text(encoding="utf-8-sig"))
        app = OpCtx(memory["config"]["dataset"], memory=memory)
        print(json.dumps(app.answer(args.question), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
