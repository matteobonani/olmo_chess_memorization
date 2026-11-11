
import argparse, json, os, re, subprocess, sys, time
import pandas as pd
import chess

from vllm import LLM, SamplingParams
import torch

from test import puzzle_position

MODELS = {

    "olmo3-32b-base":         ("allenai/Olmo-3-1125-32B",           "base"),
    "olmo31-32b-instruct":    ("allenai/Olmo-3.1-32B-Instruct",     "instruct"),
    "olmo31-32b-think":       ("allenai/Olmo-3.1-32B-Think",        "think"),


    "olmo31-32b-instruct-sft": ("allenai/Olmo-3.1-32B-Instruct-SFT", "instruct"),
    "olmo31-32b-instruct-dpo": ("allenai/Olmo-3.1-32B-Instruct-DPO", "instruct"),
    "olmo3-32b-think-sft":    ("allenai/Olmo-3-32B-Think-SFT",      "think"),
    "olmo3-32b-think-dpo":    ("allenai/Olmo-3-32B-Think-DPO",      "think"),
    "olmo3-32b-think":        ("allenai/Olmo-3-32B-Think",          "think"),
}

MAIN_MODELS = ["olmo3-32b-base", "olmo31-32b-instruct", "olmo31-32b-think"]

FORMATS = {"base": ["fen", "pgn"], "instruct": ["fen", "pgn", "pieces"], "think": ["fen", "pgn", "pieces"]}

DATASETS = ["dolma3_chess/seen_final.csv", "dolma3_chess/unseen_2000.csv"]

PROMPTS = {
    "fen": "Here is a chess position in FEN notation:\n{position}\n\n",
    "pgn": "Here is a chess game in PGN notation, up to the current position:\n{position}\n\n",
    "pieces": "Here is a chess position, given as the list of pieces on the board:\n{position}\n\n",
}
PROMPT = (
    "{intro}"
    "{side} to move. There is a move that delivers checkmate immediately (mate in one). "
    "Find it.\n\n"
    "Write your final answer on the last line in this exact form:\n"
    "Answer: <move in standard algebraic notation, e.g. Qh7#>"
)

SAN_RE = re.compile(r"(?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h](?:x[a-h])?[1-8](?:=?[QRBN])?|O-O(?:-O)?)[+#]?")
UCI_RE = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")


NAMES = [(chess.KING, "King"), (chess.QUEEN, "Queen"), (chess.ROOK, "Rook"),
         (chess.BISHOP, "Bishop"), (chess.KNIGHT, "Knight"), (chess.PAWN, "Pawn")]

def piece_list(fen):
    """List of pieces like 'White: King g1; Rook a1; Pawns f2, g2, h2'."""
    b = chess.Board(fen)
    lines = []
    for color, cname in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        parts = []
        for pt, name in NAMES:
            # take
            sqs = [chess.square_name(s) for s in sorted(b.pieces(pt, color))]

            if sqs:
                parts.append(f"{name}{'s' if len(sqs) > 1 else ''} {', '.join(sqs)}")
        lines.append(f"{cname}: " + "; ".join(parts))

    return "\n".join(lines)


def position_text(puzzle, fmt):
    """Take te corresponding position format: FEN, pieces list, PGN moves."""
    if fmt == "fen":
        return puzzle.fen

    if fmt == "pieces":
        return piece_list(puzzle.fen)

    head = f'[SetUp "1"]\n[FEN "{puzzle.start_fen}"]\n\n' if puzzle.start_fen else ""
    return head + puzzle.pgn_prefix


def base_prefix(puzzle, side, fmt):
    """Same as position_text method but for base model only"""
    if fmt == "pgn":
        return position_text(puzzle, "pgn")

    if fmt == "pieces":
        raise SystemExit("'pieces format' is only for Instruct and Think models")

    start = "1." if side == "White" else "1..."
    return f'[Event "?"]\n[SetUp "1"]\n[FEN "{puzzle.fen}"]\n\n{start}'


def parse_move(text, board, kind):
    """Extract the move from the answer. (move read, legal move, None)."""
    if kind != "base":
        # remove reasoning
        text = text.split("</think>")[-1]

        answers = re.findall(r"Answer:\s*\**\s*([^\s*]+)", text, re.I)
        candidates = answers[-1:] if answers else SAN_RE.findall(text)[-1:] + UCI_RE.findall(text)[-1:]
    else:
        candidates = SAN_RE.findall(text)[:1] + UCI_RE.findall(text)[:1]

    for c in candidates:
        c = c.strip(".,;:)").replace("0-0-0", "O-O-O").replace("0-0", "O-O")
        for parse in (board.parse_san, board.parse_uci):
            try:
                return c, parse(c)
            except ValueError:
                continue

    return (candidates[0] if candidates else ""), None


def is_mate(board, move):
    if move is None:
        return False

    board.push(move)
    check = board.is_checkmate()
    board.pop()

    return check

def run_all(args):

    models = MAIN_MODELS if args.model == "all" else [args.model]
    jobs = [(m, f) for m in models
            for f in (FORMATS[MODELS[m][1]] if args.format == "all" else [args.format])
            if f in FORMATS[MODELS[m][1]]]

    print("n run:", ", ".join(f"{m} ({f})" for m, f in jobs), flush=True)

    failed = []
    for m, f in jobs:
        cmd = [sys.executable, os.path.abspath(__file__), "--model", m, "--format", f, "--chunk", str(args.chunk)]
        for opt in ("limit", "max_tokens", "tp"):
            if getattr(args, opt) is not None:
                cmd += [f"--{opt.replace('_', '-')}", str(getattr(args, opt))]
        print(f"\n===== {m} ({f}) =====", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(f"{m} ({f})")
    print("\nall done!" + (f" Failing: {', '.join(failed)} (rerun to restart from the failed ones)" if failed else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS) + ["all"], help='a single model, "all" for the 3 principal models')
    ap.add_argument("--format", default="pgn", choices=["pgn", "fen", "pieces", "all"], help='position format input, "all" for all format')
    ap.add_argument("--limit", type=int, default=None, help="n puzzle per type")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--tp", type=int, default=None, help="how many GPU (default = every")
    ap.add_argument("--chunk", type=int, default=256, help="save per n puzzle")
    args = ap.parse_args()

    if args.model == "all" or args.format == "all":
        return run_all(args)


    repo, kind = MODELS[args.model]
    max_tokens = args.max_tokens or {"base": 12, "instruct": 1024, "think": 16384}[kind]
    tp = args.tp or max(1, torch.cuda.device_count())

    os.makedirs("results", exist_ok=True)
    out_path = f"results/{args.model}_{args.format}.jsonl"


    # puzzle
    dfs = []

    for path in DATASETS:
        df = pd.read_csv(path, keep_default_na=False)
        if args.limit:
            df = df.sample(min(args.limit, len(df)), random_state=0)
        dfs.append(df)

    puzzles = pd.concat(dfs).reset_index(drop=True)
    puzzle_done = set()

    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            puzzle_done = {json.loads(l)["puzzle_id"] for l in f}

    puzzles = puzzles[~puzzles.puzzle_id.isin(puzzle_done)]
    print(f"{args.model} ({args.format}): {len(puzzles)} puzzle to do ({len(puzzle_done)} puzzle done)", flush=True)

    if puzzles.empty:
        return

    llm = LLM(model=repo, tensor_parallel_size=tp, dtype="bfloat16",
              max_model_len=max_tokens + 1024, enable_prefix_caching=True,
              gpu_memory_utilization=0.90, seed=0)
    tok = llm.get_tokenizer()

    if kind == "think":
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.6, top_p=0.95, seed=0)
    else:
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0,
                            stop=["\n"] if kind == "base" else None)

    for start in range(0, len(puzzles), args.chunk):
        block = puzzles.iloc[start:start + args.chunk]
        boards = [chess.Board(f) for f in block.fen]
        sides = ["White" if b.turn else "Black" for b in boards]
        t0 = time.time()


        if kind == "base":
            prompts = [base_prefix(p, s, args.format) + " " for p, s in zip(block.itertuples(), sides)]
            outs = llm.generate(prompts, sp, use_tqdm=False)
        else:
            convs = [[{"role": "user", "content": PROMPT.format(
                         intro=PROMPTS[args.format].format(position=position_text(p, args.format)), side=s)}]
                     for p, s in zip(block.itertuples(), sides)]

            outs = llm.chat(convs, sp, use_tqdm=False)


        ranks = [None] * len(block)
        if kind == "base":
            reqs, owner = [], []
            for i, (p, s, b) in enumerate(zip(block.itertuples(), sides, boards)):
                prefix_ids = tok.encode(base_prefix(p, s, args.format))
                for m in b.legal_moves:

                    move_ids = tok.encode(" " + b.san(m).rstrip("+#"), add_special_tokens=False)
                    reqs.append({"prompt_token_ids": prefix_ids + move_ids})
                    owner.append((i, m, len(prefix_ids)))
            scored = llm.generate(reqs, SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
            per_puzzle = {}
            for (i, m, n_pre), o in zip(owner, scored):
                ids = o.prompt_token_ids
                lp = sum(o.prompt_logprobs[k][ids[k]].logprob for k in range(n_pre, len(ids)))
                per_puzzle.setdefault(i, []).append((lp, m))
            for i, lst in per_puzzle.items():
                lst.sort(key=lambda x: -x[0])
                order = [m for _, m in lst]
                correct = [m for m in order if is_mate(boards[i], m)]
                ranks[i] = {"rank": order.index(correct[0]) + 1 if correct else None,
                            "n_legal": len(order),
                            "top_move": boards[i].san(order[0]),
                            "logprob_correct": next(lp for lp, m in lst if m == correct[0]) if correct else None}

        # saving
        with open(out_path, "a", encoding="utf-8") as f:
            for i, (p, b, o) in enumerate(zip(block.itertuples(), boards, outs)):
                text = o.outputs[0].text
                raw, move = parse_move(text, b, kind)
                rec = {"puzzle_id": p.puzzle_id,
                       "group": "seen" if p.puzzle_id.startswith("S") else "unseen",
                       "model": args.model, "format": args.format, "kind": kind, "answer_raw": raw,
                       "answer_uci": move.uci() if move else "", "legal": move is not None,
                       "correct": is_mate(b, move),
                       "completion_tokens": len(o.outputs[0].token_ids),
                       "finish_reason": o.outputs[0].finish_reason, "response": text}
                if ranks[i]:
                    rec.update(ranks[i])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"{start + len(block)}/{len(puzzles)} puzzle ({time.time() - t0:.0f}s)", flush=True)

    res = pd.read_json(out_path, lines=True)
    print("\n", res.groupby("group")[["legal", "correct"]].mean().round(3).to_string())
    if "rank" in res:
        print("\nright move at the top:",
              res.assign(top1=res["rank"] == 1).groupby("group").top1.mean().round(3).to_dict())
    print(f"cutted answer (finish_reason=length): {(res.finish_reason == 'length').sum()}")


if __name__ == "__main__":
    main()
