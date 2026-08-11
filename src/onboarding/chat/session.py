"""One chat interface, three framework implementations.

Same system prompt, same read-only tools — only the orchestration differs, which
is the whole point of the comparison:

* **LangChain** — a single ``create_agent`` loop. Conversational by nature.
* **LangGraph** — a two-node graph (``agent`` ↔ ``tools``) with a message
  channel and a real checkpointer, so the conversation survives the process.
* **MAF** — an ``Agent`` over ``OpenAIChatCompletionClient`` with the same tools
  attached, driven turn by turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from onboarding.chat.tools import READ_ONLY_TOOLS
from onboarding.core import qa
from onboarding.core.audit import JsonlAuditSink
from onboarding.core.concepts import Concept, concept
from onboarding.core.prompts import library
from onboarding.core.schemas import CustomerRecord

MASKED_EXAMPLE = "d***@b***.com or +44***42"


@concept(Concept.CONTEXT_AWARE_PROMPT, Concept.POLICY_CONSTRAINED)
def system_prompt(record: CustomerRecord | None = None) -> tuple[str, Any]:
    """Render the shared chat system prompt, optionally pinned to one customer."""
    current = ""
    if record is not None:
        current = (
            "CURRENT CUSTOMER\n"
            f"The employee is asking about {record.primary_contact.full_name} "
            f"at {record.company_name}, on the {record.effective_plan} plan. "
            "Assume an unqualified question refers to them."
        )
    return library().render(
        "chat_analyst",
        plan_names=", ".join(qa.KNOWN_PLANS),
        masked_example=MASKED_EXAMPLE,
        current_customer=current,
    )


@dataclass
class Turn:
    question: str
    answer: str
    tool_calls: list[str] = field(default_factory=list)


class ChatSession(Protocol):
    framework: str

    async def ask(self, question: str) -> Turn: ...


# ---------------------------------------------------------------------------
# LangChain — the native conversational case
# ---------------------------------------------------------------------------


class LangChainChat:
    framework = "langchain"

    def __init__(self, record: CustomerRecord | None = None, sink: JsonlAuditSink | None = None):
        from langchain.agents import create_agent
        from langchain_core.tools import tool

        from onboarding.core.llm import make_langchain_model

        prompt, self.prompt_ref = system_prompt(record)
        self._agent = create_agent(
            model=make_langchain_model(),
            tools=[tool(fn) for fn in READ_ONLY_TOOLS],
            system_prompt=prompt,
            checkpointer=None,
        )
        self._history: list[dict[str, str]] = []
        self._sink = sink

    @concept(Concept.SINGLE_VS_MULTI_AGENT, Concept.ACTION)
    async def ask(self, question: str) -> Turn:
        self._history.append({"role": "user", "content": question})
        result = await self._agent.ainvoke({"messages": list(self._history)})
        messages = result.get("messages", [])
        answer = _last_text(messages)
        self._history.append({"role": "assistant", "content": answer})
        return Turn(question=question, answer=answer, tool_calls=_tool_names(messages))


# ---------------------------------------------------------------------------
# LangGraph — an explicit agent/tools loop with durable history
# ---------------------------------------------------------------------------


class LangGraphChat:
    framework = "langgraph"

    def __init__(self, record: CustomerRecord | None = None, sink: JsonlAuditSink | None = None):
        from langchain_core.tools import tool
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode, tools_condition

        from onboarding.core.llm import make_langchain_model

        prompt, self.prompt_ref = system_prompt(record)
        tools = [tool(fn) for fn in READ_ONLY_TOOLS]
        model = make_langchain_model().bind_tools(tools)

        async def agent(state: MessagesState) -> dict[str, Any]:
            from langchain_core.messages import SystemMessage

            reply = await model.ainvoke([SystemMessage(content=prompt), *state["messages"]])
            return {"messages": [reply]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent)
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "agent")
        # The loop that makes this a graph rather than a single call: the agent
        # goes to tools whenever it asked for one, and back again afterwards.
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        self._graph = graph.compile()
        self._history: list[Any] = []
        self._sink = sink

    @concept(Concept.CONDITIONAL_BRANCHING, Concept.ACTION)
    async def ask(self, question: str) -> Turn:
        from langchain_core.messages import HumanMessage

        self._history.append(HumanMessage(content=question))
        result = await self._graph.ainvoke({"messages": list(self._history)})
        messages = result.get("messages", [])
        self._history = list(messages)
        return Turn(question=question, answer=_last_text(messages), tool_calls=_tool_names(messages))


# ---------------------------------------------------------------------------
# Microsoft Agent Framework
# ---------------------------------------------------------------------------


class MafChat:
    framework = "maf"

    def __init__(self, record: CustomerRecord | None = None, sink: JsonlAuditSink | None = None):
        from agent_framework import Agent

        from onboarding.core.llm import make_maf_client

        prompt, self.prompt_ref = system_prompt(record)
        self._agent = Agent(
            client=make_maf_client(),
            instructions=prompt,
            tools=list(READ_ONLY_TOOLS),
        )
        self._thread: Any = None
        self._sink = sink

    @concept(Concept.ACTION, Concept.SINGLE_VS_MULTI_AGENT)
    async def ask(self, question: str) -> Turn:
        response = await self._agent.run(question, thread=self._thread)
        self._thread = getattr(response, "thread", self._thread)
        answer = str(getattr(response, "text", "") or response)
        return Turn(question=question, answer=answer, tool_calls=_maf_tool_names(response))


# ---------------------------------------------------------------------------


def build_session(
    framework: str, record: CustomerRecord | None = None, sink: JsonlAuditSink | None = None
) -> ChatSession:
    key = framework.lower()
    if key in ("lc", "langchain"):
        return LangChainChat(record, sink)
    if key in ("lg", "langgraph"):
        return LangGraphChat(record, sink)
    if key in ("maf", "agent-framework", "agent_framework"):
        return MafChat(record, sink)
    raise ValueError(f"unknown framework {framework!r}; expected maf, langchain or langgraph")


def _last_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "tool":
            continue
        content = getattr(message, "content", None)
        text = _flatten(content)
        if text.strip():
            return text.strip()
    return ""


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, str | dict)
        )
    return "" if content is None else str(content)


def _tool_names(messages: list[Any]) -> list[str]:
    names: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                names.append(name)
    return names


def _maf_tool_names(response: Any) -> list[str]:
    names: list[str] = []
    for message in getattr(response, "messages", None) or []:
        for content in getattr(message, "contents", None) or []:
            name = getattr(content, "name", None)
            if name and type(content).__name__.startswith("FunctionCall"):
                names.append(name)
    return names
