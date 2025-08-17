#!/usr/bin/env python3
"""
Day 3-7 stack for Hive LM:
- Day 3: decoder-only Transformer (policy + value), ctx=256, d=512, L=8, heads=8, mlp=2048.
- Day 4: training loop with AdamW(β=(0.9,0.95), wd=0.1), cosine LR + warmup, AMP, grad-accum.
- Day 5: legality mask from engine and shallow PUCT MCTS (N rollouts).
- Day 6: play/eval vs random and negamax heuristic; ablations and metrics.
- Day 7: resignation token, UHP I/O, top-k checkpoints, temperature sweep.
- Extras: think-time head (optional), guided data upscaling via masked sampling/MCTS.

Inputs: JSONL(.gz) with {moves:[str], result:"W"|"B"|"D"|null, optional think_sec:[float] per ply}.
Tokens are move strings seen in data + specials.
"""
from __future__ import annotations
import argparse
import dataclasses as dc
import gzip
import io
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------- Engine bridge (UHP) ---------------------------
ROOT = Path(__file__).resolve().parents[1]
src_path = str(ROOT / "src")
import sys
if src_path not in sys.path:
  sys.path.insert(0, src_path)
from core.board import Board
from core.enums import GameState
from ai.brain import AlphaBetaPruner
from engine import Engine

# ---------------------------- Tokenizer ---------------------------------
SPECIALS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "<resign>": 4}

class MoveTokenizer:
  def __init__(self, vocab: dict[str, int]):
    self.stoi = vocab
    self.itos = {i: s for s, i in vocab.items()}
    self.pad_id = SPECIALS["<pad>"]
    self.bos_id = SPECIALS["<bos>"]
    self.eos_id = SPECIALS["<eos>"]
    self.unk_id = SPECIALS["<unk>"]
    self.resign_id = SPECIALS["<resign>"]

  def encode_moves(self, moves: Sequence[str], add_bos: bool = True, add_eos: bool = True) -> list[int]:
    ids: list[int] = []
    if add_bos:
      ids.append(self.bos_id)
    for m in moves:
      ids.append(self.stoi.get(m, self.unk_id))
    if add_eos:
      ids.append(self.eos_id)
    return ids

  @staticmethod
  def build_from_jsonl(paths: list[str], min_count: int = 1, limit: Optional[int] = None) -> "MoveTokenizer":
    counts: dict[str, int] = {}
    def _open(p: str) -> io.TextIOBase:
      return io.TextIOWrapper(gzip.open(p, "rb"), encoding="utf-8") if p.endswith(".gz") else open(p, "r", encoding="utf-8")
    n = 0
    for p in paths:
      with _open(p) as fh:
        for line in fh:
          if not line.strip():
            continue
          try:
            rec = json.loads(line)
            for m in rec.get("moves", []):
              counts[m] = counts.get(m, 0) + 1
          except Exception:
            pass
          n += 1
          if limit and n >= limit:
            break
    vocab = dict(SPECIALS)
    for m, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
      if c >= min_count and m not in vocab:
        vocab[m] = len(vocab)
    return MoveTokenizer(vocab)

# ------------------------------ Dataset ---------------------------------
@dc.dataclass(slots=True)
class Ex:
  ids: list[int]
  value_targets: list[float]
  think_targets: Optional[list[float]]

class HiveJsonlDataset(Dataset[Ex]):
  def __init__(self, paths: list[str], tok: MoveTokenizer, ctx: int, split_frac: float = 0.95, train: bool = True) -> None:
    self.paths = paths
    self.tok = tok
    self.ctx = ctx
    # index lines
    self._idx: list[tuple[int, int]] = []
    rng = random.Random(123)
    for i, p in enumerate(paths):
      opener = gzip.open if p.endswith(".gz") else open
      mode = "rt" if p.endswith(".gz") else "r"
      with opener(p, mode, encoding="utf-8") as fh:
        for ln, line in enumerate(fh):
          if line.strip():
            self._idx.append((i, ln))
    rng.shuffle(self._idx)
    cut = int(len(self._idx) * split_frac)
    self.indices = self._idx[:cut] if train else self._idx[cut:]

  def __len__(self) -> int:
    return len(self.indices)

  def __getitem__(self, idx: int) -> Ex:
    path_idx, line_no = self.indices[idx]
    path = self.paths[path_idx]
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    rec: Any = {}
    with opener(path, mode, encoding="utf-8") as fh:
      for ln, line in enumerate(fh):
        if ln == line_no:
          rec = json.loads(line)
          break
    moves: list[str] = rec.get("moves", [])
    res = rec.get("result")
    r = 1.0 if res == "W" else (-1.0 if res == "B" else 0.0)
    ids = self.tok.encode_moves(moves, add_bos=True, add_eos=True)
    targets: list[float] = []
    for t in range(len(ids)):
      side = 1.0 if (t % 2 == 0) else -1.0
      targets.append(r * side if res is not None else 0.0)
    think = rec.get("think_sec")
    think_targets = None
    if isinstance(think, list):
      think_targets = [float(x) for x in think[:len(ids)]] # type: ignore
    if len(ids) > self.ctx:
      ids = ids[: self.ctx]
      targets = targets[: self.ctx]
      if think_targets is not None:
        think_targets = think_targets[: self.ctx]
    return Ex(ids=ids, value_targets=targets, think_targets=think_targets)

# ------------------------------ Collate ---------------------------------
@dc.dataclass(slots=True)
class Batch:
  x: torch.Tensor
  mask: torch.Tensor
  y_next: torch.Tensor
  v_tgt: torch.Tensor
  t_tgt: Optional[torch.Tensor]

def collate(exs: list[Ex], pad_id: int, ctx: int) -> Batch:
  T = min(ctx, max(len(e.ids) for e in exs))
  B = len(exs)
  x = torch.full((B, T), pad_id, dtype=torch.long)
  y = torch.full((B, T), -100, dtype=torch.long)
  v = torch.zeros((B, T), dtype=torch.float32)
  m = torch.zeros((B, T), dtype=torch.bool)
  t: Optional[torch.Tensor] = None
  have_t = any(e.think_targets is not None for e in exs)
  if have_t:
    t = torch.zeros((B, T), dtype=torch.float32)
  for i, e in enumerate(exs):
    n = min(T, len(e.ids))
    x[i, :n] = torch.tensor(e.ids[:n])
    y_seq = e.ids[1:n] + [pad_id] if n > 0 else []
    y[i, :n] = torch.tensor(y_seq + [-100] * (n - len(y_seq)))
    v[i, :n] = torch.tensor(e.value_targets[:n])
    if have_t and t is not None:
      if e.think_targets is not None:
        t[i, :n] = torch.tensor(e.think_targets[:n])
    m[i, :n] = True
  return Batch(x=x, mask=m, y_next=y, v_tgt=v, t_tgt=t)

# ------------------------------- Model ----------------------------------
class TransformerBlock(nn.Module):
  def __init__(self, d: int, heads: int, mlp: int, attn_drop: float, resid_drop: float) -> None:
    super().__init__() # type: ignore
    self.ln1 = nn.LayerNorm(d)
    self.attn = nn.MultiheadAttention(d, heads, dropout=attn_drop, batch_first=True)
    self.ln2 = nn.LayerNorm(d)
    self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))
    self.drop = nn.Dropout(resid_drop)

  def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    h = self.ln1(x)
    a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
    x = x + self.drop(a)
    h = self.ln2(x)
    x = x + self.drop(self.mlp(h))
    return x

class PolicyValueTransformer(nn.Module):
  def __init__(self, vocab_size: int, ctx: int = 256, d: int = 512, L: int = 8, heads: int = 8, mlp: int = 2048, attn_drop: float = 0.0, resid_drop: float = 0.0, think_head: bool = True) -> None:
    super().__init__() # type: ignore
    self.ctx = ctx
    self.tok_emb = nn.Embedding(vocab_size, d)
    self.pos_emb = nn.Embedding(ctx, d)
    self.blocks = nn.ModuleList([TransformerBlock(d, heads, mlp, attn_drop, resid_drop) for _ in range(L)])
    self.ln_f = nn.LayerNorm(d)
    self.head_policy = nn.Linear(d, vocab_size, bias=False)
    self.head_value = nn.Linear(d, 1)
    self.head_think = nn.Linear(d, 1) if think_head else None

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    T = x.size(1)
    pos = torch.arange(T, device=x.device).unsqueeze(0)
    h = self.tok_emb(x) + self.pos_emb(pos)
    causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
    for blk in self.blocks:
      h = blk(h, attn_mask=causal)
    h = self.ln_f(h)
    logits = self.head_policy(h)
    v = torch.tanh(self.head_value(h)).squeeze(-1)
    t = None
    if self.head_think is not None:
      t = torch.nn.functional.softplus(self.head_think(h)).squeeze(-1)  # seconds ≥ 0
    return logits, v, t

# --------------------------- Legality mask -------------------------------
class HiveEnv:
  def __init__(self) -> None:
    self.b = Board()
  def reset(self, gamestring: str = "") -> None:
    self.b = Board(gamestring) if gamestring else Board()
  def legal_moves(self) -> list[str]:
    ms = self.b.calculate_valid_moves()
    return sorted(self.b.stringify_move(m) for m in ms) if ms else ["pass"]
  def play(self, move: str) -> None:
    self.b.play(move)
  def is_over(self) -> bool:
    return bool(getattr(self.b, "gameover", False))
  def result(self) -> Optional[str]:
    st = getattr(self.b, "state", None)
    if st is None:
      return None
    if st is GameState.DRAW:
      return "D"
    if st is GameState.WHITE_WINS:
      return "W"
    if st is GameState.BLACK_WINS:
      return "B"
    return None

def mask_logits_with_legals(logits: torch.Tensor, legal_ids: list[int]) -> torch.Tensor:
  if not legal_ids:
    return logits
  mask = torch.full_like(logits, float("-inf"))
  mask[..., legal_ids] = logits[..., legal_ids]
  return mask

# ------------------------------- MCTS -----------------------------------
@dc.dataclass(slots=True)
class MCTSConf:
  n: int = 128
  cpuct: float = 1.5
  dirichlet_alpha: float = 0.3
  root_noise: float = 0.25
  temperature_until: int = 20

class MCTS:
  def __init__(self, model: PolicyValueTransformer, tok: MoveTokenizer, conf: MCTSConf, device: str) -> None:
    self.model = model
    self.tok = tok
    self.conf = conf
    self.device = device

  def _policy_value(self, env: HiveEnv, prefix_ids: list[int]) -> tuple[torch.Tensor, float, list[int]]:
    x = torch.tensor(prefix_ids, dtype=torch.long, device=self.device).unsqueeze(0)
    with torch.no_grad():
      logits, v, _ = self.model(x)
    legals = env.legal_moves()
    legal_ids = [self.tok.stoi.get(m, self.tok.unk_id) for m in legals]
    masked = mask_logits_with_legals(logits[:, -1, :], legal_ids)
    probs = torch.softmax(masked, dim=-1).squeeze(0)
    return probs, float(v[0, -1].item()), legal_ids

  def select_move(self, env: HiveEnv, gamestring: str, temperature: float = 1.0) -> str:
    # simple PUCT with in-memory tree per root
    env.reset(gamestring)
    prefix_ids = [self.tok.bos_id]
    # Node stats keyed by move-id
    N: dict[int, int] = {}
    W: dict[int, float] = {}
    P: Optional[torch.Tensor]
    probs, _v_root, legal_ids = self._policy_value(env, prefix_ids)
    P = probs.clone()
    # add Dirichlet noise at root
    if len(legal_ids) > 1 and self.conf.root_noise > 0:
      noise = torch.distributions.Dirichlet(torch.full((len(P),), self.conf.dirichlet_alpha)).sample().to(P)
      P = (1 - self.conf.root_noise) * P + self.conf.root_noise * noise

    # Rollouts
    for _ in range(self.conf.n):
      # select best a maximizing Q + U
      best_a = None
      best_score = -1e9
      sumN = sum(N.get(i, 0) for i in range(len(P))) + 1
      for aid in range(len(P)):
        p = float(P[aid].item())
        n = N.get(aid, 0)
        q = W.get(aid, 0.0) / n if n > 0 else 0.0
        u = self.conf.cpuct * p * math.sqrt(sumN) / (1 + n)
        s = q + u
        if s > best_score:
          best_score, best_a = s, aid
      assert best_a is not None
      # simulate 1-ply: apply chosen move and eval leaf
      move_id = best_a
      move_str = self.tok.itos.get(legal_ids[move_id], "<unk>")
      env2 = HiveEnv(); env2.reset(gamestring)
      env2.play(move_str)
      # leaf eval
      ids2 = [self.tok.bos_id]
      _, v2, _ = self._policy_value(env2, ids2)
      # backup
      N[best_a] = N.get(best_a, 0) + 1
      W[best_a] = W.get(best_a, 0.0) + v2

    # pick move
    counts = torch.tensor([N.get(i, 0) for i in range(len(P))], dtype=torch.float32)
    if len(counts) == 0 or counts.sum() == 0:
      # fallback to policy
      probs = torch.softmax(P, dim=-1)
      choice = torch.multinomial(probs.pow(1.0/temperature), 1).item()
    else:
      if temperature > 0:
        choice = torch.multinomial((counts + 1e-6).pow(1.0/temperature), 1).item()
      else:
        choice = torch.argmax(counts).item()
    chosen_move = self.tok.itos.get(legal_ids[choice], "<unk>") # type: ignore
    return chosen_move

# ------------------------------ Training --------------------------------
@dc.dataclass(slots=True)
class TrainArgs:
  data: list[str]
  out_dir: str
  ctx: int
  d: int
  L: int
  heads: int
  mlp: int
  batch_size: int
  lr: float
  wd: float
  epochs: int
  lambda_value: float
  lambda_think: float
  seed: int
  min_count: int
  vocab: Optional[str]
  device: str
  warmup_steps: int
  grad_accum: int
  topk: int


def parse_train_args(argv: Optional[Iterable[str]] = None) -> TrainArgs:
  p = argparse.ArgumentParser(description="Day 3–7 training")
  p.add_argument("--data", nargs="+", type=str, required=True)
  p.add_argument("--out-dir", type=str, default="runs/day3")
  p.add_argument("--ctx", type=int, default=256)
  p.add_argument("--d", type=int, default=512)
  p.add_argument("--L", type=int, default=8)
  p.add_argument("--heads", type=int, default=8)
  p.add_argument("--mlp", type=int, default=2048)
  p.add_argument("--batch-size", type=int, default=64)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--wd", type=float, default=0.1)
  p.add_argument("--epochs", type=int, default=3)
  p.add_argument("--lambda-value", type=float, default=0.25)
  p.add_argument("--lambda-think", type=float, default=0.0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--min-count", type=int, default=1)
  p.add_argument("--vocab", type=str, default="")
  p.add_argument("--device", type=str, default="auto")
  p.add_argument("--warmup-steps", type=int, default=500)
  p.add_argument("--grad-accum", type=int, default=1)
  p.add_argument("--topk", type=int, default=3)
  a = p.parse_args(list(argv) if argv is not None else None)
  return TrainArgs(
    data=a.data, out_dir=a.out_dir, ctx=a.ctx, d=a.d, L=a.L, heads=a.heads, mlp=a.mlp,
    batch_size=a.batch_size, lr=a.lr, wd=a.wd, epochs=a.epochs,
    lambda_value=a.lambda_value, lambda_think=a.lambda_think, seed=a.seed,
    min_count=a.min_count, vocab=(a.vocab or None), device=a.device,
    warmup_steps=a.warmup_steps, grad_accum=a.grad_accum, topk=a.topk,
  )


def count_params(m: nn.Module) -> int:
  return sum(p.numel() for p in m.parameters())


def train_main(argv: Optional[Iterable[str]] = None) -> int:
  args = parse_train_args(argv)
  os.makedirs(args.out_dir, exist_ok=True)
  torch.manual_seed(args.seed) # type: ignore
  random.seed(args.seed)

  # tokenizer
  if args.vocab and os.path.exists(args.vocab):
    with open(args.vocab, "r", encoding="utf-8") as f:
      vocab = json.load(f)
    tok = MoveTokenizer(vocab)
  else:
    tok = MoveTokenizer.build_from_jsonl(args.data, min_count=args.min_count)
    with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as f:
      json.dump(tok.stoi, f, ensure_ascii=False, indent=2)

  # data
  train_ds = HiveJsonlDataset(args.data, tok, ctx=args.ctx, train=True)
  val_ds = HiveJsonlDataset(args.data, tok, ctx=args.ctx, train=False)
  coll = lambda b: collate(b, tok.pad_id, args.ctx)
  train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll, num_workers=2)
  val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll, num_workers=2)

  # model
  device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
  model = PolicyValueTransformer(vocab_size=len(tok.stoi), ctx=args.ctx, d=args.d, L=args.L, heads=args.heads, mlp=args.mlp, think_head=(args.lambda_think > 0))
  model.to(device)
  print(f"params={count_params(model)/1e6:.1f}M, vocab={len(tok.stoi)}")

  # opt + sched
  opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
  total_steps = max(1, (len(train_ld) * args.epochs + args.grad_accum - 1) // args.grad_accum)
  def lr_at(step: int) -> float:
    if step < args.warmup_steps:
      return args.lr * step / max(1, args.warmup_steps)
    t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
    return 0.5 * args.lr * (1 + math.cos(math.pi * t))
  scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda")) # type: ignore
  ce = nn.CrossEntropyLoss(ignore_index=-100)
  mse = nn.MSELoss(reduction="none")

  topk: list[tuple[float, str]] = []
  step = 0
  for ep in range(1, args.epochs + 1):
    model.train(True)
    accum = 0
    opt.zero_grad(set_to_none=True)
    tr_lm = tr_v = tr_t = 0.0; seen = 0
    for batch in train_ld:
      x = batch.x.to(device); y = batch.y_next.to(device)
      v_tgt = batch.v_tgt.to(device); m = batch.mask.to(device)
      t_tgt = batch.t_tgt.to(device) if (batch.t_tgt is not None) else None
      for g in opt.param_groups:
        g["lr"] = lr_at(step)
      with torch.amp.autocast('cuda', enabled=(device == "cuda")): # type: ignore
        logits, v, t = model(x)
        lm = ce(logits.view(-1, logits.size(-1)), y.view(-1))
        v_loss_full = mse(v, v_tgt)
        v_loss = (v_loss_full[m].mean()) if m.any() else torch.tensor(0.0, device=device)
        if t is not None and t_tgt is not None:
          t_loss_full = mse(t, t_tgt)
          t_loss = (t_loss_full[m].mean()) if m.any() else torch.tensor(0.0, device=device)
        else:
          t_loss = torch.tensor(0.0, device=device)
        loss = lm + args.lambda_value * v_loss + args.lambda_think * t_loss
      scaler.scale(loss / args.grad_accum).backward() # type: ignore
      accum += 1
      if accum >= args.grad_accum:
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        accum = 0; step += 1
      tr_lm += float(lm.detach()) * x.size(0); tr_v += float(v_loss.detach()) * x.size(0); tr_t += float(t_loss.detach()) * x.size(0)
      seen += x.size(0)
    # val
    model.train(False)
    va_lm = va_v = va_t = 0.0; vseen = 0
    with torch.no_grad():
      for batch in val_ld:
        x = batch.x.to(device); y = batch.y_next.to(device)
        v_tgt = batch.v_tgt.to(device); m = batch.mask.to(device)
        t_tgt = batch.t_tgt.to(device) if (batch.t_tgt is not None) else None
        logits, v, t = model(x)
        lm = ce(logits.view(-1, logits.size(-1)), y.view(-1))
        v_loss_full = mse(v, v_tgt); v_loss = (v_loss_full[m].mean()) if m.any() else torch.tensor(0.0, device=device)
        if t is not None and t_tgt is not None:
          t_loss_full = mse(t, t_tgt); t_loss = (t_loss_full[m].mean()) if m.any() else torch.tensor(0.0, device=device)
        else:
          t_loss = torch.tensor(0.0, device=device)
        va_lm += float(lm) * x.size(0); va_v += float(v_loss) * x.size(0); va_t += float(t_loss) * x.size(0)
        vseen += x.size(0)
    tr_lm/=max(1,seen); tr_v/=max(1,seen); tr_t/=max(1,seen)
    va_lm/=max(1,vseen); va_v/=max(1,vseen); va_t/=max(1,vseen)
    va_tot = va_lm + args.lambda_value*va_v + args.lambda_think*va_t
    print(f"epoch {ep}: train lm {tr_lm:.3f} v {tr_v:.3f} t {tr_t:.3f} | val lm {va_lm:.3f} v {va_v:.3f} t {va_t:.3f} tot {va_tot:.3f}")
    # save + top-k
    ckpt: dict[str, dict[str, Any]] = {"args": dataclasses.asdict(args) if (dataclasses := dc) else {}, "model_state": model.state_dict(), "vocab": tok.stoi}
    path = os.path.join(args.out_dir, f"ep{ep:02d}.pt"); torch.save(ckpt, path)
    topk.append((va_tot, path)); topk.sort(key=lambda x: x[0])
    for i, (_, p) in enumerate(topk[: args.topk]):
      tgt = os.path.join(args.out_dir, f"top{i+1}.pt")
      if os.path.abspath(p) != os.path.abspath(tgt):
        try: torch.save(torch.load(p, map_location="cpu"), tgt)
        except Exception: pass
    if va_tot <= min(x for x,_ in topk[: args.topk]):
      torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
  return 0

# ------------------------------- Play/Eval ------------------------------
def load_model(ckpt_path: str, device: str = "auto") -> tuple[PolicyValueTransformer, MoveTokenizer, str]:
  c = torch.load(ckpt_path, map_location="cpu")
  tok = MoveTokenizer(c["vocab"])  # type: ignore[index]
  a = c.get("args", {})
  model = PolicyValueTransformer(vocab_size=len(tok.stoi), ctx=a.get("ctx",256), d=a.get("d",512), L=a.get("L",8), heads=a.get("heads",8), mlp=a.get("mlp",2048), think_head=(a.get("lambda_think",0.0)>0))
  model.load_state_dict(c["model_state"])  # type: ignore[index]
  dev = ("cuda" if torch.cuda.is_available() else "cpu") if device=="auto" else device
  model.to(dev).eval()
  return model, tok, dev

class LMHiveAgent:
  def __init__(self, model: PolicyValueTransformer, tok: MoveTokenizer, device: str, use_mcts: bool = True, mcts_n: int = 128, temperature: float = 1.0, resign_thresh: float = -0.9) -> None:
    self.model = model; self.tok = tok; self.device = device
    self.use_mcts = use_mcts; self.temperature = temperature
    self.mcts = MCTS(model, tok, MCTSConf(n=mcts_n), device) if use_mcts else None
    self.resign_thresh = resign_thresh

  def select(self, env: HiveEnv, gamestring: str) -> str:
    # resignation rule
    x = torch.tensor([[self.tok.bos_id]], dtype=torch.long, device=self.device)
    with torch.no_grad():
      _, v, _ = self.model(x)
    if float(v[0,-1].item()) < self.resign_thresh:
      return "pass"  # no explicit resign move in UHP; choose no-op here
    if self.use_mcts and self.mcts is not None:
      return self.mcts.select_move(env, gamestring, self.temperature)
    # masked sampling
    env.reset(gamestring)
    legals = env.legal_moves(); legal_ids = [self.tok.stoi.get(m, self.tok.unk_id) for m in legals]
    with torch.no_grad():
      logits, _, _ = self.model(torch.tensor([[self.tok.bos_id]], dtype=torch.long, device=self.device))
      masked = mask_logits_with_legals(logits[:, -1, :], legal_ids)
      probs = torch.softmax(masked / max(1e-4,self.temperature), dim=-1).squeeze(0)
      choice = torch.multinomial(probs, 1).item()
      return self.tok.itos.get(legal_ids[choice], "<unk>")

# Simple eval loop
class RandomAgent:
  def move(self, env: HiveEnv) -> str:
    return random.choice(env.legal_moves())

class NegamaxAgent:
  def __init__(self, movetime_s: int = 1) -> None:
    self.time = movetime_s
    self._brain = AlphaBetaPruner()
  def move(self, env: HiveEnv) -> str:
    b = Board(str(env.b))  # copy via gamestring
    mv = self._brain.find_best_move(b, Engine.DEFAULT_MAX_BRANCHING_FACTOR, time_limit=self.time)
    return mv if mv in env.legal_moves() else random.choice(env.legal_moves())

@dc.dataclass(slots=True)
class EvalConf:
  games: int = 100
  mcts_n: int = 128
  temperature: float = 1.0
  negamax_time: int = 1

@dc.dataclass(slots=True)
class EvalResult:
  win: int; loss: int; draw: int; avg_len: float

def play_match(agent_w: Any, agent_b: Any, max_plies: int = 300) -> tuple[str, int]:
  env = HiveEnv()
  env.reset("")
  plies = 0
  for ply in range(max_plies):
    if env.is_over():
      break
    move = agent_w.move(env) if ply % 2 == 0 else agent_b.move(env)
    env.play(move)
    plies += 1
  res = env.result() or "D"
  return res, plies

def eval_model(ckpt: str, vs: str = "random", conf: EvalConf = EvalConf()) -> EvalResult:
  model, tok, dev = load_model(ckpt)
  lm_agent = LMHiveAgent(model, tok, dev, use_mcts=(conf.mcts_n>0), mcts_n=conf.mcts_n, temperature=conf.temperature)
  if vs == "random":
    opp = RandomAgent()
  else:
    opp = NegamaxAgent(conf.negamax_time)
  W = L = D = 0
  total_len=0
  for g in range(conf.games):
    aw, ab = (lm_agent, opp) if g % 2==0 else (opp, lm_agent)
    res, length = play_match(aw, ab)
    if res == "W":
      W += 1 if g % 2==0 else 0
      L += 0 if g % 2==0 else 1
    elif res == "B":
      L += 1 if g % 2==0 else 0
      W += 0 if g % 2==0 else 1
    else:
      D += 1
    total_len += length
  return EvalResult(W, L, D, total_len / max(1, conf.games))

# -------------------------- Guided Upscaling ----------------------------
@dc.dataclass(slots=True)
class UpscaleArgs:
  ckpt: str
  out: str
  games: int
  mcts_n: int
  temperature: float


def upscale_main(argv: Optional[Iterable[str]] = None) -> int:
  p = argparse.ArgumentParser(description="Guided data upscaling with current model")
  p.add_argument("--ckpt", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--games", type=int, default=10000)
  p.add_argument("--mcts-n", type=int, default=64)
  p.add_argument("--temperature", type=float, default=1.0)
  a = p.parse_args(list(argv) if argv is not None else None)
  model, tok, dev = load_model(a.ckpt)
  agent = LMHiveAgent(model, tok, dev, use_mcts=(a.mcts_n>0), mcts_n=a.mcts_n, temperature=a.temperature)
  env = HiveEnv()
  # write JSONL(.gz)
  out = a.out; os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
  fh: io.TextIOBase = io.TextIOWrapper(gzip.open(out, "wb"), encoding="utf-8") if out.endswith(".gz") else open(out, "w", encoding="utf-8")
  for _g in range(a.games):
    env.reset("")
    moves: list[str] = []
    for _ply in range(300):
      if env.is_over(): break
      mv = agent.select(env, str(env.b))
      env.play(mv); moves.append(mv)
    rec: dict[str, list[str] | str | int | dict[str, str] | None] = {"moves": moves, "result": env.result(), "ply_count": len(moves), "meta": {"source": "upscale"}}
    fh.write(json.dumps(rec) + "\n")
  fh.close()
  return 0

# ------------------------------ CLI glue --------------------------------
if __name__ == "__main__":
  ap = argparse.ArgumentParser(description="Day3–7 Stack")
  sub = ap.add_subparsers(dest="cmd", required=True)

  tr = sub.add_parser("train", help="train the model")
  # reuse TrainArgs
  for flag, typ, default in [
    ("--data", str, None), ("--out-dir", str, "runs/day3"), ("--ctx", int, 256), ("--d", int, 512), ("--L", int, 8),
    ("--heads", int, 8), ("--mlp", int, 2048), ("--batch-size", int, 64), ("--lr", float, 3e-4), ("--wd", float, 0.1),
    ("--epochs", int, 3), ("--lambda-value", float, 0.25), ("--lambda-think", float, 0.0), ("--seed", int, 42),
    ("--min-count", int, 1), ("--vocab", str, ""), ("--device", str, "auto"), ("--warmup-steps", int, 500),
    ("--grad-accum", int, 1), ("--topk", int, 3)
  ]:
    if flag == "--data":
      tr.add_argument(flag, nargs="+", required=True)
    else:
      tr.add_argument(flag, type=typ, default=default)

  pl = sub.add_parser("play", help="single move from UHP gamestring")
  pl.add_argument("--ckpt", required=True)
  pl.add_argument("--gamestring", type=str, default="")
  pl.add_argument("--mcts-n", type=int, default=128)
  pl.add_argument("--temperature", type=float, default=1.0)

  ev = sub.add_parser("eval", help="evaluate vs random or negamax")
  ev.add_argument("--ckpt", required=True)
  ev.add_argument("--vs", choices=["random", "negamax"], default="random")
  ev.add_argument("--games", type=int, default=200)
  ev.add_argument("--mcts-n", type=int, default=128)
  ev.add_argument("--temperature", type=float, default=1.0)
  ev.add_argument("--negamax-time", type=float, default=1.0)

  up = sub.add_parser("upscale", help="guided data generation")
  up.add_argument("--ckpt", required=True)
  up.add_argument("--out", required=True)
  up.add_argument("--games", type=int, default=10000)
  up.add_argument("--mcts-n", type=int, default=64)
  up.add_argument("--temperature", type=float, default=1.0)

  args, rest = ap.parse_known_args()
  if args.cmd == "train":
    raise SystemExit(train_main(rest))
  elif args.cmd == "play":
    model, tok, dev = load_model(args.ckpt)
    env = HiveEnv()
    agent = LMHiveAgent(model, tok, dev, use_mcts=(args.mcts_n>0), mcts_n=args.mcts_n, temperature=args.temperature)
    mv = agent.select(env, args.gamestring)
    print(mv)
  elif args.cmd == "eval":
    r = eval_model(args.ckpt, vs=args.vs, conf=EvalConf(games=args.games, mcts_n=args.mcts_n, temperature=args.temperature, negamax_time=args.negamax_time))
    print(json.dumps(dc.asdict(r)))
  elif args.cmd == "upscale":
    raise SystemExit(upscale_main(rest))
