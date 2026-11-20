"""Read-only skills offered to the language model as native tool calls.

The regex layer (``SkillManager``) handles explicit commands. Everything else reaches the model with these tools so it can
use the whole conversation to decide, for example that a bare "VTI" or "and Tesla?" is a follow-up price question. Tools
here never change anything on the machine; actions with side effects (apps, volume, deleting notes) keep the spoken yes.
"""
from __future__ import annotations

import re

from .base import FAILED, SkillContext, SkillResult
from .mac_control import FOLDERS

# skill id -> OpenAI-style tool schema. The descriptions double as the instructions the model reads.
TOOL_SCHEMAS = {
    "web_search": [
        {"type": "function", "function": {
            "name": "web_search",
            "description": "Search the web for current or changing information: news, weather, sports results, events, "
                           "recent facts. Do not use for chat, opinions or timeless knowledge you already have.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Self-contained search query; use names from the conversation."},
                "news": {"type": "boolean", "description": "True to search news articles instead of web pages."}},
                "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "stock_quote",
            "description": "Get the live price of a stock or ETF. Prices change every second, so ALWAYS call this for any "
                           "price question, even if a price was mentioned earlier in the conversation.",
            "parameters": {"type": "object", "properties": {
                "company": {"type": "string", "description": "Company name or ticker, e.g. VTI or Amazon."}},
                "required": ["company"]}}},
    ],
    "mac_control": [
        {"type": "function", "function": {
            "name": "list_folder",
            "description": "List the file names in the user's Desktop, Documents or Downloads folder.",
            "parameters": {"type": "object", "properties": {
                "folder": {"type": "string", "enum": sorted(FOLDERS)}}, "required": ["folder"]}}},
    ],
}


# Wording that clearly asks for live data. On such turns the model is not allowed to answer in words without a tool result:
# if it tries, the code discards that answer and searches with the user's own sentence instead.
LIVE_CUE_RE = re.compile(r"\b(?:price|prices|trading|worth|stocks?|etfs?|shares|tickers?|markets?|scores?|weather|forecast|news|"
                         r"headlines|latest|going on with|happening|who won|exchange rate)\b", re.I)


def tool_schemas(enabled: set[str]) -> list[dict]:
    """Tool schemas for the enabled skills."""
    return [tool for skill_id, tools in TOOL_SCHEMAS.items() if skill_id in enabled for tool in tools]


async def call_tool(skills: dict, enabled: set[str], name: str, args: dict, ctx: SkillContext) -> SkillResult:
    """Run one model-requested tool through the owning skill; bad names or arguments become an honest FAILED result."""
    try:
        if name == "web_search" and "web_search" in enabled:
            query = str(args.get("query", "")).strip()[:120]
            skill_args = {"query": query, "news": bool(args.get("news"))}
            skill_id = "web_search"
            if len(query) < 2:
                raise ValueError("empty query")
        elif name == "stock_quote" and "web_search" in enabled:
            company = str(args.get("company", "")).strip()[:60]
            if not company:
                raise ValueError("empty company")
            skill_args, skill_id = {"stock": company}, "web_search"
        elif name == "list_folder" and "mac_control" in enabled:
            folder = str(args.get("folder", "")).lower()
            if folder not in FOLDERS:
                raise ValueError(f"folder must be one of {sorted(FOLDERS)}")
            skill_args, skill_id = {"action": "list", "folder": folder}, "mac_control"
        else:
            raise ValueError(f"tool '{name}' is not available")
        result = await skills[skill_id].run(skill_args, ctx)
    except Exception as error:
        result = SkillResult(FAILED, f"The tool call failed ({error}). Do not answer from memory; tell the user briefly.",
                             f"Tool failed: {error}")
        skill_id = name
    result.skill = skill_id
    return result
