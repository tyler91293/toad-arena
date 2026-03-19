#!/usr/bin/env python3
"""
Toad — MoltBet Arena agent (Railway deployment).

Self-contained Flask server.
Primary LLM: Groq (OpenAI-compatible, runs Qwen — free tier, fast).
Fallback: Anthropic Haiku (if Groq fails or not configured).

Env vars:
  GROQ_API_KEY      — Groq API key (primary, free at console.groq.com)
  ANTHROPIC_API_KEY — Anthropic fallback
  PORT              — set automatically by Railway
"""
import json
import logging
import os
import random
import re

import anthropic
from openai import OpenAI
from flask import Flask, jsonify, request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="[toad] %(message)s")
log = logging.getLogger("toad")

# Primary: Groq (Qwen) — free, fast
GROQ_KEY      = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL    = os.environ.get("GROQ_MODEL", "qwen-qwq-32b")
groq_client   = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_KEY else None

# Fallback: Anthropic Haiku
ANTHROPIC_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL  = "claude-haiku-4-5-20251001"
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None


def llm(prompt: str, max_tokens: int = 150) -> str:
    """Call Groq (Qwen) primary, fall back to Anthropic Haiku."""
    if groq_client:
        try:
            r = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"Groq failed: {e} — falling back to Anthropic")

    if anthropic_client:
        msg = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()

    raise RuntimeError("No LLM configured")


def parse_json(text: str) -> dict:
    """Extract JSON from LLM response, tolerating markdown fences."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ── Chess ──────────────────────────────────────────────────────────────────────

def handle_chess(data: dict) -> dict:
    legal = data.get("legal_moves") or []
    if not legal:
        return {"move": "resign", "comment": "Nothing left."}

    prompt = f"""You are Toad, a competitive chess AI. Play strong, talk trash.

Position (FEN): {data.get("fen", "")}
You are: {data.get("your_color", "white")}
Recent moves: {" ".join((data.get("move_history") or [])[-8:]) or "none"}
Opponent: {(data.get("opponent") or {}).get("name", "?")} ELO {(data.get("opponent") or {}).get("elo", 1200)}
Legal moves: {", ".join(legal)}

Pick the strongest move. Reply ONLY with valid JSON (no markdown):
{{"move": "<exact SAN from legal_moves>", "comment": "<trash talk, max 80 chars>"}}

The move MUST exactly match one entry from legal_moves."""

    try:
        text = llm(prompt, max_tokens=120)
        result = parse_json(text)
        move    = result.get("move", "")
        comment = str(result.get("comment", ""))[:100]
        if move not in legal:
            log.warning(f"invalid move '{move}' — random fallback")
            return {"move": random.choice(legal), "comment": "Recalculating..."}
        return {"move": move, "comment": comment}
    except Exception as e:
        log.error(f"chess error: {e}")
        return {"move": random.choice(legal), "comment": "Trust the process."}


# ── Poker ──────────────────────────────────────────────────────────────────────

def handle_poker(data: dict) -> dict:
    actions = data.get("actions") or data.get("legal_moves") or ["fold"]
    hole    = data.get("hole_cards") or []
    community = data.get("community_cards") or []
    pot     = data.get("pot", 0)

    prompt = f"""You are Toad, a calculating poker AI. Play smart, mix it up.

Hole cards: {hole}
Community: {community}
Pot: {pot}
Available actions: {actions}

Pick the best action. Reply ONLY with valid JSON:
{{"move": "<exact action from list>", "comment": "<short trash talk>"}}

The move MUST exactly match one entry from: {actions}"""

    try:
        text   = llm(prompt, max_tokens=100)
        result = parse_json(text)
        move   = result.get("move", "")
        if move not in actions:
            for preferred in ("check", "call", "raise"):
                if preferred in actions:
                    return {"move": preferred, "comment": "Pot odds favor the patient."}
            move = actions[0]
        return {"move": move, "comment": str(result.get("comment", ""))[:80]}
    except Exception as e:
        log.error(f"poker error: {e}")
        for preferred in ("check", "call"):
            if preferred in actions:
                return {"move": preferred, "comment": "Playing it safe."}
        return {"move": actions[0], "comment": "..."}


# ── Connect 4 ──────────────────────────────────────────────────────────────────

def handle_connect4(data: dict) -> dict:
    legal = data.get("legal_moves") or data.get("actions") or []
    board = data.get("board", "")

    prompt = f"""You are Toad, playing Connect 4. Play to win.

Board state: {board}
You are: {data.get("your_color", data.get("your_piece", "?"))}
Legal columns to drop: {legal}

Pick the best column. Reply ONLY with valid JSON:
{{"move": "<column from legal_moves>", "comment": "<short trash talk>"}}

The move MUST exactly match one entry from: {legal}"""

    try:
        text   = llm(prompt, max_tokens=80)
        result = parse_json(text)
        move   = str(result.get("move", ""))
        if move not in [str(m) for m in legal]:
            return {"move": str(random.choice(legal)), "comment": "Dropping heat."}
        return {"move": move, "comment": str(result.get("comment", ""))[:80]}
    except Exception as e:
        log.error(f"connect4 error: {e}")
        return {"move": str(random.choice(legal)), "comment": "..."}


# ── Checkers ───────────────────────────────────────────────────────────────────

def handle_checkers(data: dict) -> dict:
    legal = data.get("legal_moves") or []
    if not legal:
        return {"move": "pass", "comment": "No moves."}

    prompt = f"""You are Toad, playing Checkers. Play aggressively, prioritize jumps.

Board: {data.get("board", "")}
You are: {data.get("your_color", "?")}
Legal moves: {legal}

Pick the best move. Reply ONLY with valid JSON:
{{"move": "<exact move from legal_moves>", "comment": "<short trash talk>"}}

The move MUST exactly match one entry from: {legal}"""

    try:
        text   = llm(prompt, max_tokens=80)
        result = parse_json(text)
        move   = result.get("move", "")
        if move not in legal:
            return {"move": random.choice(legal), "comment": "Jumping ahead."}
        return {"move": move, "comment": str(result.get("comment", ""))[:80]}
    except Exception as e:
        log.error(f"checkers error: {e}")
        return {"move": random.choice(legal), "comment": "..."}


# ── Othello ────────────────────────────────────────────────────────────────────

def handle_othello(data: dict) -> dict:
    legal = data.get("legal_moves") or []
    if not legal:
        return {"move": "pass", "comment": "Passing."}

    prompt = f"""You are Toad, playing Othello/Reversi. Prioritize corners and edges.

Board: {data.get("board", "")}
You are: {data.get("your_color", "?")}
Legal moves: {legal}

Pick the best square. Reply ONLY with valid JSON:
{{"move": "<exact square from legal_moves>", "comment": "<short trash talk>"}}

The move MUST exactly match one entry from: {legal}"""

    try:
        text   = llm(prompt, max_tokens=80)
        result = parse_json(text)
        move   = result.get("move", "")
        if move not in legal:
            return {"move": random.choice(legal), "comment": "Flipping the script."}
        return {"move": move, "comment": str(result.get("comment", ""))[:80]}
    except Exception as e:
        log.error(f"othello error: {e}")
        return {"move": random.choice(legal), "comment": "..."}


# ── Liar's Dice ────────────────────────────────────────────────────────────────

def handle_liars_dice(data: dict) -> dict:
    actions = data.get("actions") or data.get("legal_moves") or ["challenge"]
    dice    = data.get("dice") or data.get("my_dice") or []
    bid     = data.get("current_bid") or data.get("bid") or {}
    players = data.get("players_dice_count") or {}

    prompt = f"""You are Toad, playing Liar's Dice. Bluff boldly, challenge wisely.

My dice: {dice}
Current bid: {bid}
Players (dice count): {players}
Available actions: {actions}

Pick the best action. Reply ONLY with valid JSON:
{{"move": "<exact action from actions>", "comment": "<short trash talk>"}}

If bidding, the move should be one of the available actions exactly.
The move MUST exactly match one entry from: {actions}"""

    try:
        text   = llm(prompt, max_tokens=100)
        result = parse_json(text)
        move   = result.get("move", "")
        if move not in actions:
            return {"move": random.choice(actions), "comment": "Calculated risk."}
        return {"move": move, "comment": str(result.get("comment", ""))[:80]}
    except Exception as e:
        log.error(f"liars-dice error: {e}")
        return {"move": random.choice(actions), "comment": "..."}


# ── Debate ─────────────────────────────────────────────────────────────────────

def handle_debate(data: dict) -> dict:
    topic    = data.get("topic", "the topic")
    position = data.get("your_position") or data.get("position") or data.get("side", "")
    history  = data.get("argument_history") or data.get("history") or []
    round_n  = data.get("round", 1)

    prompt = f"""You are Toad, an AI debater. Argue sharply, cite evidence, be provocative.

Topic: {topic}
Your position: {position}
Round: {round_n}
Previous arguments: {json.dumps(history[-4:]) if history else "none"}

Write a compelling argument (2-3 sentences max). Be specific and confident.
Reply ONLY with valid JSON:
{{"move": "<your argument text>", "comment": "<one-liner taunt>"}}"""

    try:
        text   = llm(prompt, max_tokens=200)
        result = parse_json(text)
        return {
            "move":    str(result.get("move", "The evidence speaks for itself."))[:500],
            "comment": str(result.get("comment", ""))[:100],
        }
    except Exception as e:
        log.error(f"debate error: {e}")
        return {"move": f"The case for {position} is unassailable.", "comment": "Next."}


# ── Trivia ─────────────────────────────────────────────────────────────────────

def handle_trivia(data: dict) -> dict:
    question = data.get("question", "")
    choices  = data.get("choices") or data.get("options") or data.get("legal_moves") or []
    category = data.get("category", "")

    if not choices:
        prompt = f"""Trivia question: {question}\nAnswer briefly and directly."""
        try:
            answer = llm(prompt, max_tokens=60)
            return {"move": answer, "comment": "Too easy."}
        except Exception:
            return {"move": "A", "comment": "..."}

    prompt = f"""You are Toad, answering a trivia question. Answer correctly.

Category: {category}
Question: {question}
Choices: {choices}

Pick the correct answer. Reply ONLY with valid JSON:
{{"move": "<exact answer from choices>", "comment": "<short quip>"}}

The move MUST exactly match one entry from: {choices}"""

    try:
        text   = llm(prompt, max_tokens=100)
        result = parse_json(text)
        move   = result.get("move", "")
        if move not in choices:
            return {"move": random.choice(choices), "comment": "Educated guess."}
        return {"move": move, "comment": str(result.get("comment", ""))[:80]}
    except Exception as e:
        log.error(f"trivia error: {e}")
        return {"move": random.choice(choices), "comment": "..."}


# ── Prisoner's Dilemma ─────────────────────────────────────────────────────────

def handle_prisoners_dilemma(data: dict) -> dict:
    actions = data.get("actions") or data.get("legal_moves") or ["cooperate", "defect"]
    history = data.get("history") or []
    round_n = data.get("round", 1)

    # Tit-for-tat with occasional defection
    if history:
        last = history[-1]
        opp_last = last.get("opponent") or last.get("other")
        if opp_last == "defect":
            move = "defect"
            comment = "You played yourself."
        else:
            move = "cooperate" if random.random() > 0.15 else "defect"
            comment = "Keeping the peace." if move == "cooperate" else "Surprise."
    else:
        move    = "cooperate"
        comment = "Starting friendly. For now."

    if move not in actions:
        move = random.choice(actions)

    return {"move": move, "comment": comment}


# ── Auction ────────────────────────────────────────────────────────────────────

def handle_auction(data: dict) -> dict:
    actions     = data.get("actions") or data.get("legal_moves") or ["pass"]
    item        = data.get("item") or data.get("item_name", "the item")
    current_bid = data.get("current_bid") or data.get("highest_bid") or 0
    budget      = data.get("budget") or data.get("my_budget") or 1000
    round_n     = data.get("round", 1)

    prompt = f"""You are Toad, bidding in an auction. Bid strategically.

Item: {item}
Current highest bid: {current_bid}
Your budget: {budget}
Available actions: {actions}
Round: {round_n}

Pick the best action. Reply ONLY with valid JSON:
{{"move": "<exact action from actions>", "comment": "<short quip>"}}

The move MUST exactly match one entry from: {actions}"""

    try:
        text   = llm(prompt, max_tokens=100)
        result = parse_json(text)
        move   = result.get("move", "")
        if move not in actions:
            return {"move": random.choice(actions), "comment": "Calculating value."}
        return {"move": move, "comment": str(result.get("comment", ""))[:80]}
    except Exception as e:
        log.error(f"auction error: {e}")
        return {"move": random.choice(actions), "comment": "..."}


# ── Battleground ───────────────────────────────────────────────────────────────

def handle_battleground(data: dict) -> dict:
    moves = data.get("legal_moves") or data.get("actions") or data.get("moves") or ["wait"]

    prompt = f"""You are Toad, in a territory battle. Play aggressively to dominate.

Game state: {json.dumps({k: v for k, v in data.items() if k not in ("game_id",)})[:800]}
Available moves: {moves}

Pick the best move. Reply ONLY with valid JSON:
{{"move": "<exact move from list>", "comment": "<short trash talk>"}}

The move MUST exactly match one entry from: {moves}"""

    try:
        text   = llm(prompt, max_tokens=100)
        result = parse_json(text)
        move   = result.get("move", "")
        if move not in moves:
            return {"move": random.choice(moves), "comment": "Advancing."}
        return {"move": move, "comment": str(result.get("comment", ""))[:80]}
    except Exception as e:
        log.error(f"battleground error: {e}")
        return {"move": random.choice(moves), "comment": "..."}


# ── Router ─────────────────────────────────────────────────────────────────────

GAME_HANDLERS = {
    "chess":               handle_chess,
    "poker":               handle_poker,
    "connect4":            handle_connect4,
    "checkers":            handle_checkers,
    "othello":             handle_othello,
    "liars-dice":          handle_liars_dice,
    "liars_dice":          handle_liars_dice,
    "debate":              handle_debate,
    "trivia":              handle_trivia,
    "prisoners-dilemma":   handle_prisoners_dilemma,
    "prisoners_dilemma":   handle_prisoners_dilemma,
    "auction":             handle_auction,
    "battleground":        handle_battleground,
}


def detect_game(data: dict) -> str:
    if gt := data.get("game_type"):
        return gt
    if "fen" in data or ("legal_moves" in data and "your_color" in data):
        return "chess"
    if "hole_cards" in data or "community_cards" in data:
        return "poker"
    if "question" in data:
        return "trivia"
    if "topic" in data:
        return "debate"
    if "dice" in data or "my_dice" in data:
        return "liars-dice"
    return "battleground"


@app.route("/", methods=["POST"])
def agent():
    data    = request.get_json(force=True, silent=True) or {}
    game    = detect_game(data)
    game_id = data.get("game_id", "?")
    opp     = (data.get("opponent") or {}).get("name", "?")
    log.info(f"game={game} id={game_id[:8]} opp={opp}")

    handler = GAME_HANDLERS.get(game, handle_battleground)
    result  = handler(data)
    log.info(f"move={result.get('move')!r}")
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "agent": "Toad", "games": list(GAME_HANDLERS.keys())})


@app.route("/", methods=["GET"])
def root_get():
    return jsonify({"status": "ok", "agent": "Toad"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    log.info(f"Toad starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
