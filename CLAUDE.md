# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Purpose

This repository is an internal AI education and learning hub for Blink, a software development company. It is **not a product codebase** — its output is teaching material (tutorials, guides, examples, exercises) about using AI tools effectively at work.

## Audience

Content in this repo serves two distinct audiences and most material should be written with this split in mind:

- **Engineering (~35 developers)** — comfortable with CLIs, code, APIs, and technical depth. Tutorials for this group can assume git, shell, package managers, and language fundamentals.
- **Non-engineering staff (HR, Ops, CEO, Sales, etc.)** — should not need to read code or use a terminal to get value. Tutorials for this group should lead with outcomes and use GUI-based tools, web apps, or no-code workflows wherever possible.

When writing or reviewing material, identify which audience it targets and match the assumed prior knowledge accordingly. If a piece is meant for both, separate the technical deep-dive from the high-level walkthrough rather than mixing them.

## Structure

- `everyone/` — tutorials for non-technical staff. No code or terminal assumed.
- `engineering/` — tutorials for engineers.
- `prompts/` — shared prompt library across both audiences.

Each tutorial is a single Markdown file with embedded images, code snippets, and exercises. Avoid creating new directories or files unless the content volume justifies it.

## When proposing changes

- Keep it simple. The repo is intentionally lightweight; default to fewer directories, fewer files, fewer abstractions.
- Don't propose subdirectories under `everyone/` or `engineering/` until there is enough content to justify them.
- For engineering content, check whether it's enabling the Transformation Plan (BEIF workflows, the 4 context artifacts, the 5-command pipeline) before treating it as generic "AI for devs."
- For everyone content, do not borrow BEIF terminology — that audience is outside the engineering plan.
