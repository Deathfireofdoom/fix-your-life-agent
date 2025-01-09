from typing import Annotated
from typing_extensions import TypedDict
import json
import re

from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from agent.prompts import SYSTEM_PROMPT, generate_insights_prompt, context_extraction_prompt
from agent.services.principle import PrincipleService

from agent.TEMPORARY import TAGS, USE_CASES  # TODO: remove this later

class State(TypedDict):
    messages: Annotated[list, add_messages]
    next_node: str
    context: dict[str, str]

class Agent:
    def __init__(self, principle_service: PrincipleService, llm):
        self.system_prompt = SystemMessage(content=SYSTEM_PROMPT)
        self.graph = self._init_graph()
        self.principle_service = principle_service
        self.llm = llm

    def ask(self, prompt: str):
        for _ in self.graph.stream({"messages": [("user", prompt)]}):
            pass

    def _init_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("entrypoint", self._entrypoint)
        
        graph_builder.add_node("add-book", self._add_book)
        
        graph_builder.add_node("context-gather", self._generate_actions_context_gather)
        graph_builder.add_node("input-stage", self._input_stage)
        graph_builder.add_node("generate-actions", self._generate_actions)

        # Conditional edges
        graph_builder.add_edge(START, "entrypoint")
        graph_builder.add_conditional_edges(
            "entrypoint",
            lambda result: result["next_node"].lower(),
            {"add-book": "add-book", "generate-actions": "context-gather", END: END}
        )

        graph_builder.add_conditional_edges(
            "context-gather",
            lambda state: state.get("next_node", "input-stage"),
            {"input-stage": "input-stage", "generate-actions": "generate-actions"}
        )

        graph_builder.add_edge(
            "input-stage",
            "context-gather"
        )

        graph_builder.add_edge(
            "generate-actions",
            END
        )

        return graph_builder.compile()

    def _entrypoint(self, state: State):
        last_message = state["messages"][-1].content.lower()
        if "add book" in last_message:
            return {"next_node": "add-book"}
        elif "generate actions" in last_message:
            return {"next_node": "generate-actions"}
        else:
            return {"next_node": END}

    def _add_book(self, state: State):
        last_message = state["messages"][-1]
        book_title = last_message.content.replace("add book", "").strip()

        prompt = generate_insights_prompt(book_title=book_title, tags=TAGS, use_cases=USE_CASES)

        response = self.llm.invoke(
            [
                self.system_prompt,
                HumanMessage(content=prompt)
            ]
        )

        # Use PrincipleService to save the insights
        self.principle_service.add_principles_from_llm_response(response_content=response.content, book_title=book_title)

        return {"messages": [("system", f"Book '{book_title}' has been added.")], "next_node": END}

    def _generate_actions_context_gather(self, state: State):
        last_message = state["messages"][-1].content

        context = state.get("context", {})
        print(context)
        system_prompt = context_extraction_prompt.format(current_context=json.dumps(context, indent=2))

        llm_response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Current context: {context}. User said: '{last_message}'. What context fields are still needed? Return your response as a JSON object with a 'question' key if more context is needed, or 'complete': true if the context is sufficient.")
        ])

        response = self._parse_llm_response_context_gather(llm_response.content)
        context.update(response.get("new_context", {}))
        
        if response.get("question") and len(context) < 5:
            return {
                "messages": [("system", response["question"])],
                "next_node": "input-stage",
                "context": context
            }

        return {
            "messages": [("system", "Thank you! Let's generate some actionable insights.")],
            "next_node": "generate-actions",
            "context": context
        }

    def _parse_llm_response_context_gather(self, llm_response_content: str) -> dict:
        """
        Parses the LLM response to dynamically extract new context fields and follow-up questions.

        Args:
            llm_response_content (str): The response content from the LLM as a string.

        Returns:
            dict: A dictionary containing the parsed new context fields and any follow-up question.
        """
        try:
            # Remove any backticks or markdown-style code blocks
            cleaned_response = re.sub(r"```.*?\n|```", "", llm_response_content, flags=re.DOTALL)

            print(cleaned_response)

            # Attempt to parse the cleaned response as JSON
            response = json.loads(cleaned_response)

            # Validate the expected keys in the response
            if not isinstance(response, dict) or ("complete" not in response and "question" not in response):
                raise ValueError("Invalid LLM response format.")

            # Extract new context fields and follow-up question
            new_context = response.get("new_context", {})
            question = response.get("question", None)
            complete = response.get("complete", False)

            return {
                "new_context": new_context,
                "question": question,
                "complete": complete
            }

        except json.JSONDecodeError as e:
            print(f"Error decoding LLM response as JSON: {e}")
            return {
                "new_context": {},
                "question": "Sorry, I couldn't understand your response. Could you clarify?",
                "complete": False
            }

    def _input_stage(self, state: State):
        # Extract the latest system message
        last_system_message = state["messages"][-1].content

        # Print the system's question and wait for user input
        user_input = input(f"{last_system_message}\n> ")

        # Return the user input as a new message
        return {"messages": [("user", user_input)], "next_node": "context-gather"}
    
    def _generate_actions(self, state: State):
        context = state.get("context", {})

        print(context)
        prompt = f"""
        You are an assistant generating actionable insights based on user context.

        Given the context: {json.dumps(context, indent=2)}

        Select the most relevant tags and use cases from the following lists:
        - Tags: {", ".join(TAGS)}
        - Use Cases: {", ".join(USE_CASES)}

        Return your response as raw JSON in the following format:
        {{
            "tags": ["Tag1", "Tag2"],
            "use_cases": ["UseCase1", "UseCase2"]
        }}
    """

        print(prompt)
        # Ask the LLM for relevant tags and use cases
        llm_response = self.llm.invoke([
            SystemMessage(content="You are an assistant generating actionable insights based on user context."),
            HumanMessage(content=prompt)
        ])

        print(llm_response)
        # Parse the LLM's response for tags and use cases
        llm_suggestions = self._parse_llm_response_for_tags_and_use_cases(llm_response.content)

        # Query the database for relevant principles
        actions = self.principle_service.get_principles_by_use_case_and_tags(
            use_cases=llm_suggestions["use_cases"],
            tags=llm_suggestions["tags"]
        )

        # Select 5 tips that work well together
        tips = self._select_tips(actions, context)

        # Return the tips to the user
        action_messages = [
            ("system", f"Tip {i+1}: {tip["name"]} - {tip["details"]}")
            for i, tip in enumerate(tips)
        ]

        for message in action_messages:
            print(message)
    
        return {"messages": action_messages, "next_node": END}
    
    def _parse_llm_response_for_tags_and_use_cases(self, llm_response_content: str) -> dict:
        """
        Parses the LLM response to extract tags and use cases.

        Args:
            llm_response_content (str): The response content from the LLM as a string.

        Returns:
            dict: A dictionary containing the selected tags and use cases.
        """
        try:
            cleaned_response = re.sub(r"```.*?\n|```", "", llm_response_content, flags=re.DOTALL)

            response = json.loads(cleaned_response)

            tags = response.get("tags", [])
            use_cases = response.get("use_cases", [])

            tags = [tag for tag in tags if tag in TAGS]
            use_cases = [use_case for use_case in use_cases if use_case in USE_CASES]

            return {
                "tags": tags,
                "use_cases": use_cases
            }
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            return {"tags": [], "use_cases": []}
    
    def _select_tips(self, actions: list, context: dict) -> list:
        """
        Selects 5 relevant tips from the list of actions based on the current context.

        Args:
            actions (list): A list of principles retrieved from the database.
            context (dict): The gathered context for the task.

        Returns:
            list: A list of up to 5 selected principles.
        """
        # Prepare the list of actions to pass to the LLM
        actions_list = [
            {"name": action.name, "details": action.details}
            for action in actions
        ]

        # Prepare the LLM prompt
        prompt = f"""
        You are an assistant helping a user prepare for a meeting.

        Given the following context:
        {json.dumps(context, indent=2)}

        And the following available tips:
        {json.dumps(actions_list, indent=2)}

        Select 5 tips that are most relevant to the user's context. Return the response as a JSON array in the following format:
        [
            {{"name": "Tip1", "details": "Details for Tip1"}},
            {{"name": "Tip2", "details": "Details for Tip2"}},
            ...
        ]

        Return your response as **raw JSON**.
        """

        # Query the LLM
        llm_response = self.llm.invoke([
            SystemMessage(content="You are an assistant selecting relevant tips."),
            HumanMessage(content=prompt)
        ])

        # Parse the LLM's response
        try:
            cleaned_response = re.sub(r"```.*?\n|```", "", llm_response.content, flags=re.DOTALL)
            tips = json.loads(cleaned_response)
            return tips
        
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            return actions[:5]  # Fallback to return the first 5 tips if parsing fails
