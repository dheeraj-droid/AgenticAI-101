"""One chat interface, four framework implementations.

Same system prompt, same read-only tools — only the orchestration differs, which
is the whole point of the comparison:

* **LangChain** — a single ``create_agent`` loop. Conversational by nature.
* **LangGraph** — a two-node graph (``agent`` ↔ ``tools``) with a message
  channel, so the tool loop is an explicit cycle rather than an implicit one.
* **MAF** — an ``Agent`` over ``OpenAIChatCompletionClient`` with the same tools
  attached, driven turn by turn.
* **CrewAI** — a single ``Agent`` in a one-task crew per question. A crew is
  built to complete a task and then finish, so unlike the other three it has no
  native turn loop; history is replayed into each question by hand.
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
# LangGraph — an explicit agent/tools loop
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
        # MAF carries conversation history in a session object, created once and
        # passed to every run — the framework's equivalent of the message list
        # the other two thread through by hand.
        self._session = self._agent.create_session()
        self._sink = sink

    @concept(Concept.ACTION, Concept.SINGLE_VS_MULTI_AGENT)
    async def ask(self, question: str) -> Turn:
        response = await self._agent.run(question, session=self._session)
        answer = str(getattr(response, "text", "") or response)
        return Turn(question=question, answer=answer, tool_calls=_maf_tool_names(response))


# ---------------------------------------------------------------------------
# CrewAI — one agent, one task, one crew per question
# ---------------------------------------------------------------------------


class CrewChat:
    framework = "crew"

    def __init__(self, record: CustomerRecord | None = None, sink: JsonlAuditSink | None = None):
        from onboarding.core.telemetry import opt_out

        opt_out()

        from crewai.tools import tool

        from onboarding.core.llm import make_crew_llm

        prompt, self.prompt_ref = system_prompt(record)
        self._prompt = prompt
        self._llm = make_crew_llm()
        self._tools = [tool(fn.__name__)(fn) for fn in READ_ONLY_TOOLS]
        # A crew is built to finish a task, not to hold a conversation, so the
        # transcript is carried here and replayed into each question. That is the
        # cost of the multi-agent model on a conversational workload.
        self._history: list[Turn] = []
        self._sink = sink

    @concept(Concept.ACTION, Concept.SINGLE_VS_MULTI_AGENT)
    async def ask(self, question: str) -> Turn:
        import asyncio

        from crewai import Agent, Crew, Process, Task

        analyst = Agent(
            role="Customer registry analyst",
            goal="Answer questions about onboarded customers using only the tools provided.",
            backstory=self._prompt,
            tools=self._tools,
            llm=self._llm,
            allow_delegation=False,
            verbose=False,
        )
        task = Task(
            description=f"{self._transcript()}Answer this question: {question}",
            expected_output="A short, direct answer grounded in what the tools returned.",
            agent=analyst,
        )
        crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False)
        output = await asyncio.to_thread(crew.kickoff)

        answer = str(getattr(output, "raw", "") or output).strip()
        turn = Turn(question=question, answer=answer, tool_calls=_crew_tool_names(output))
        self._history.append(turn)
        return turn

    def _transcript(self) -> str:
        if not self._history:
            return ""
        lines = ["Earlier in this conversation:"]
        for turn in self._history[-5:]:
            lines.append(f"  Q: {turn.question}")
            lines.append(f"  A: {turn.answer}")
        return "\n".join(lines) + "\n\n"


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
    if key in ("crew", "crewai"):
        return CrewChat(record, sink)
    raise ValueError(
        f"unknown framework {framework!r}; expected maf, langchain, langgraph or crew"
    )


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


def _crew_tool_names(output: Any) -> list[str]:
    """Pull tool names out of a CrewOutput.

    CrewAI reports tool usage per task rather than per message, so the names come
    off ``tasks_output`` instead of a message list.
    """
    names: list[str] = []
    for task_output in getattr(output, "tasks_output", None) or []:
        for entry in getattr(task_output, "tools_used", None) or []:
            name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
            if name:
                names.append(str(name))
    return names


def _maf_tool_names(response: Any) -> list[str]:
    """Pull tool names out of a MAF response.

    Content items are discriminated by a ``type`` field ("function_call"),
    not by their Python class, so match on that.
    """
    names: list[str] = []
    for message in getattr(response, "messages", None) or []:
        for content in getattr(message, "contents", None) or []:
            if getattr(content, "type", None) != "function_call":
                continue
            name = getattr(content, "name", None)
            if name:
                names.append(name)
    return names
