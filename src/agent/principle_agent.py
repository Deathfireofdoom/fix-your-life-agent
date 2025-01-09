import csv
import json
from typing import Annotated
from typing_extensions import TypedDict
from io import StringIO

from langchain.schema import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from agent.models import Principle, Tag, UseCase


GUIDE_LINES = """
    * Actionable that can be applied in real life
    * Precise and concise    
    """

class State(TypedDict):
    messages: Annotated[list, add_messages]
    next_node: str
    context: dict[str, str]
    principles: list[Principle]
    principles_to_self_evaluate: list[Principle]
    principles_to_user_evaluate: list[Principle]
    principles_to_adjust: list[Principle]
    reasons_for_adjustment: dict[str, str]

    # id is principle id and value is reason for adjustment
    prev_node: str


class PrincipleAgent:
    def __init__(self, llm, session):
        self.llm = llm
        self.session = session
        self.graph = self._init_graph()

    def ask(self, query):
        for r in self.graph.stream({"messages": [("user", query)]}):
            print(r)

    def _init_graph(self):
        graph_builder = StateGraph(State)
        
        # nodes
        graph_builder.add_node('extract-principles', self._extract_principles)
        graph_builder.add_node('self-evaluate-principles', self._self_evaluate_principles)
        graph_builder.add_node('adjust-principles', self._adjust_principles)
        graph_builder.add_node('user-evaluate-principles', self._user_evaluate_principles)
        graph_builder.add_node('save-principles', self._save_princples)

        # edges
        graph_builder.add_edge(START, 'extract-principles')
        graph_builder.add_edge('extract-principles', 'self-evaluate-principles')

        graph_builder.add_conditional_edges('self-evaluate-principles',
                                            lambda result: "adjust-principles" if result["principles_to_adjust"] else "user-evaluate-principles",
                                           {"adjust-principles": "adjust-principles", "user-evaluate-principles": "user-evaluate-principles"}
                                           ) 

        graph_builder.add_conditional_edges('adjust-principles',
                                            lambda result: result["next_node"],
                                             {"self-evaluate-principles": "self-evaluate-principles", "user-evaluate-principles": "user-evaluate-principles"}
                                            )
        
        graph_builder.add_conditional_edges('user-evaluate-principles',
                                            lambda result: "adjust-principles" if result["principles_to_adjust"] else "save-principles",
                                                {"adjust-principles": "adjust-principles", "save-principles": "save-principles"}
                                            )
        
        graph_builder.add_edge('save-principles', END)

        return graph_builder.compile()

    def _extract_principles(self, state: State):
        # TODO: this will later be dynamic and the book will be in the state already
        last_message = state["messages"][-1]
        book_title = last_message.content.replace("add book", "").strip()

        system_prompt: SystemMessage = self._extract_principles_system_prompt()
        human_prompt: HumanMessage = self._extract_principles_human_prompt(book_title=book_title)

        llm_response = self.llm.invoke(
            [
                system_prompt,
                human_prompt,
            ]
        )

        principles = self._extract_principles_parse_llm_response(llm_response_content=llm_response.content, book_title=book_title)
        if len(principles) == 0:
            return {
                "next_node": END
            }
        
        # debug
        print("Extracted principles:")
        for principle in principles:
            print(principle.name, principle.details)

        return {
            "principles_to_self_evaluate": principles,
            "next_node": "self_evaluate_principles",
            "prev_node": "extract-principles"
        }

    def _self_evaluate_principles(self, state: State):
        principles = state["principles_to_self_evaluate"]
        
        system_prompt: SystemMessage = self._self_evaluate_principles_system_prompt()
        human_prompt: HumanMessage = self._self_evaluate_principles_human_prompt(principles=principles)

        llm_response = self.llm.invoke(
            [
                system_prompt,
                human_prompt,
            ]
        )

        response = self._self_evaluate_principles_parse_llm_response(llm_response_content=llm_response.content)
        principles_to_adjust_ids = [int(p[0]) for p in response]
        print("Principles to adjust:", response)

        # These principles passed the self evaluation step
        principles_to_user_evaluate = state.get("principles_to_user_evaluate", [])
        
        # These principles need to be adjusted
        principles_to_adjust = []
        for principle in principles:
            if principle.id in principles_to_adjust_ids:
                principles_to_adjust.append(principle)
            else:
                principles_to_user_evaluate.append(principle)

        # reasons for adjustment
        reasons_for_adjustment = {int(p[0]): p[1] for p in response}

        return {
            "principles_to_adjust": principles_to_adjust,
            "reasons_for_adjustment": reasons_for_adjustment,
            "principles_to_self_evaluate": [],
            "principles_to_user_evaluate": principles_to_user_evaluate,
            "prev_node": "self-evaluate-principles"
        }
    
    def _adjust_principles(self, state: State):
        principles_to_adjust = state["principles_to_adjust"]
        reasons_for_adjustment = state["reasons_for_adjustment"]

        system_prompt: SystemMessage = self._adjust_principles_system_prompt()
        human_prompt: HumanMessage = self._adjust_principles_human_prompt(principles=principles_to_adjust, reasons_for_adjustment=reasons_for_adjustment)

        llm_response = self.llm.invoke(
            [
                system_prompt,
                human_prompt,
            ]
        )

        print("Adjusting principles response:", llm_response.content)
        adjusted_principles = self._adjust_principles_parse_llm_response(llm_response_content=llm_response.content)

        # debug
        print("Adjusted principles:")
        for principle in adjusted_principles:
            print(principle.name, principle.details)

        if state["prev_node"] == "self-evaluate-principles":
            state["principles_to_self_evaluate"] = adjusted_principles
        elif state["prev_node"] == "user-evaluate-principles":
            state["principles_to_user_evaluate"] = adjusted_principles

        return {
            "principles_to_adjust": [],
            "principles_to_self_evaluate": state["principles_to_self_evaluate"],
            "principles_to_user_evaluate": state["principles_to_user_evaluate"],
            "next_node": state["prev_node"],
        }

    def _user_evaluate_principles(self, state: State):
        user_principles = state["principles_to_user_evaluate"]
        for principle in user_principles:
            print(principle.name, principle.details)

        return {
            "prev_node": "user-evaluate-principles"
        }

    def _save_princples(self, state: State):
        print("Saving principles to database")

    def _extract_principles_system_prompt(self) -> SystemMessage:
        prompt = f"""
        You are a helpful assistant that extracts actionable insights (principles) from a book.

        The principles should be:
        {GUIDE_LINES}

        Return the output in the following CSV format:
        Name,Details,Tags,Use Cases

        Ensure that each field is enclosed with double quotes so it can be parsed with a csv parser.

        Example:
        Name,Details,Tags,Use Cases
        Use Mirroring,Repeat the last few words to build rapport and encourage them to elaborate.,"Negotiation","Work Meetings, Sales Calls"
        Label Emotions,Identify and verbalize the emotions to diffuse tension.,"Emotional Intelligence","Difficult Conversations"
        """
        print("Extracting principles system prompt")
        print(prompt)
        return SystemMessage(content=prompt)

    def _extract_principles_human_prompt(self, book_title: str) -> HumanMessage:
        prompt = f"""
        Extract actionable insights (principles) from the book "{book_title}".
        """
        print("Extracting principles human prompt")
        print(prompt)
        return HumanMessage(content=prompt)

    def _extract_principles_parse_llm_response(self, llm_response_content: str, book_title: str) -> list[Principle]:
        f = StringIO(llm_response_content)
        reader = csv.DictReader(f) 

        principles = []
        for row in reader:
            principle = Principle(
                book_title=book_title,
                name=row.get("Name", "").strip(),
                details=row.get("Details", "").strip(),
            )

            # TODO: check if it is an issue to add principle directly to database
            self.session.add(principle)

            # TODO: do not let the llm create tags without checking
            tags = [tag.strip() for tag in row.get("Tags", "").split(",") if tag]
            for tag_name in tags:
                tag = self._get_or_create_tag(tag_name)
                principle.tags.append(tag)

            # TODO: do not let the llm create use_cases without checking
            use_cases = [use_case.strip() for use_case in row.get("Use Cases", "").split(",") if use_case]
            for use_case_name in use_cases:
                use_case = self._get_or_create_use_case(use_case_name)
                principle.use_cases.append(use_case)
            
            principles.append(principle)
        
        self.session.commit()
        return principles

    def _self_evaluate_principles_system_prompt(self) -> SystemMessage:
        prompt = f"""
        You are a helpful assistant tasked with evaluating the principles extracted from the book.

        The principles should be:
        {GUIDE_LINES}

        If a principle needs adjustment, provide the reason for adjustment. 
        If it doesn't need adjustment, don't return the principle id in the CSV.

        Return the output in the following CSV format:
        Principle ID,Reason for Adjustment

        Ensure that each field is enclosed with double quotes so it can be parsed with a csv parser.

        Example:
        Principle ID,Reason for Adjustment
        1,"Not actionable in real life, too vague"
        4,"Not concise"
        """
        print("Self evaluating principles system prompt")
        print(prompt)
        return SystemMessage(content=prompt)


    def _self_evaluate_principles_human_prompt(self, principles: list[Principle]) -> HumanMessage:
        principles_list = [
            {
                "id": principle.id,
                "name": principle.name,
                "details": principle.details,
            }
            for principle in principles
        ]

        prompt = f"""
        Evaluate the following principles and identify if any need improvement:

        {json.dumps(principles_list, indent=2)}
        """
        print("Self evaluating principles human prompt")
        print(prompt)
        return HumanMessage(content=prompt)
    
    def _self_evaluate_principles_parse_llm_response(self, llm_response_content: str) -> list[tuple[str, str]]:
        """
        Parses the LLM response in CSV format and returns a list of tuples.
        Each tuple contains the principle ID and the reason for adjustment.

        Args:
            llm_response_content (str): The LLM's response in CSV format.

        Returns:
            list[tuple[str, str]]: A list of (principle_id, reason) tuples.
        """
        f = StringIO(llm_response_content)
        reader = csv.DictReader(f)

        principles_to_adjust = []
        for row in reader:
            principle_id = row.get("Principle ID", "").strip()
            reason = row.get("Reason for Adjustment", "").strip()

            # Skip empty rows
            if principle_id and reason:
                principles_to_adjust.append((principle_id, reason))

        return principles_to_adjust
    
    def _adjust_principles_system_prompt(self) -> SystemMessage:
        prompt = f"""
        You are a helpful assistant tasked with adjusting the principles that need improvement.

        The principles should be:
        {GUIDE_LINES}

        You will be provided with the principles that need adjustment and the reason for adjustment.
        
        Return all principles, including the adjusted ones, in the following CSV format:
        id,Name,Details,Tags,Use Cases

        Ensure that each field is enclosed with double quotes so it can be parsed with a csv parser.

        Example:
        id,Name,Details,Tags,Use Cases
        23,Use Mirroring,Repeat the last few words to build rapport and encourage them to elaborate.,"Negotiation","Work Meetings, Sales Calls"
        328,Label Emotions,Identify and verbalize the emotions to diffuse tension.,"Emotional Intelligence","Difficult Conversations"
        """
        print("Adjusting principles system prompt")
        print(prompt)
        return SystemMessage(content=prompt)

    def _adjust_principles_human_prompt(self, principles: list[Principle], reasons_for_adjustment: dict[str, str]) -> HumanMessage:
        principles_and_reason = [
            {
                "id": principle.id,
                "name": principle.name,
                "details": principle.details,
                "reason": reasons_for_adjustment.get(principle.id, "")
            }
            for principle in principles
        ]
        prompt = f"""
        Adjust the following principles based on the feedback provided:

        {json.dumps(principles_and_reason, indent=2)}
        """
        print("Adjusting principles human prompt")
        print(prompt)
        return HumanMessage(content=prompt)

    def _adjust_principles_parse_llm_response(self, llm_response_content: str) -> list[Principle]:
        """
        Parses the LLM response in CSV format and updates existing principles in the database.

        Args:
            llm_response_content (str): The LLM's response in CSV format.
            session (Session): SQLAlchemy session for database operations.

        Returns:
            List[Principle]: A list of updated Principle instances.
        """
        f = StringIO(llm_response_content)
        reader = csv.DictReader(f)

        updated_principles = []

        for row in reader:
            # Get the principle ID from the row
            principle_id = row.get("id", "").strip()

            # Fetch the principle from the database
            principle = self.session.query(Principle).get(principle_id)
            if not principle:
                print(f"Principle with ID {principle_id} not found. Skipping.")
                continue

            # Update the principle's attributes
            principle.name = row.get("Name", principle.name).strip()
            principle.details = row.get("Details", principle.details).strip()

            # Handle tags
            tags = [tag.strip() for tag in row.get("Tags", "").split(",") if tag]
            principle.tags.clear()  # Clear existing tags
            for tag_name in tags:
                tag = self._get_or_create_tag(tag_name)
                principle.tags.append(tag)

            # Handle use cases
            use_cases = [use_case.strip() for use_case in row.get("Use Cases", "").split(",") if use_case]
            principle.use_cases.clear()  # Clear existing use cases
            for use_case_name in use_cases:
                use_case = self._get_or_create_use_case(use_case_name)
                principle.use_cases.append(use_case)

            # Add the updated principle to the list
            updated_principles.append(principle)

        # Commit the changes to the database
        self.session.commit()

        return updated_principles
 
    # TODO: they should be moved
    def _get_or_create_tag(self, tag_name: str) -> Tag:
        """
        Retrieves an existing Tag or creates a new one if it doesn't exist.

        Args:
            tag_name (str): The name of the tag.

        Returns:
            Tag: The Tag model instance.
        """
        tag = self.session.query(Tag).filter_by(name=tag_name).first()
        if not tag:
            tag = Tag(name=tag_name)
            self.session.add(tag)
            self.session.commit()
        return tag

    def _get_or_create_use_case(self, use_case_name: str) -> UseCase:
        """
        Retrieves an existing UseCase or creates a new one if it doesn't exist.

        Args:
            use_case_name (str): The name of the use case.

        Returns:
            UseCase: The UseCase model instance.
        """
        use_case = self.session.query(UseCase).filter_by(name=use_case_name).first()
        if not use_case:
            use_case = UseCase(name=use_case_name)
            self.session.add(use_case)
            self.session.commit()
        return use_case