import csv
from io import StringIO
from typing import List, Optional
from agent.models import Principle, Tag, UseCase
from sqlalchemy.orm import Session

class PrincipleService:
    def __init__(self, session: Session):
        self.session = session

    def add_principles_from_llm_response(self, response_content: str, book_title: str) -> List[Principle]:
        """
        Parses the LLM response content in CSV format and saves the principles to the database.

        Args:
            response_content (str): The LLM's response in CSV format.
            book_title (str): The title of the book the principles are extracted from.

        Returns:
            List[Principle]: A list of saved Principle instances.
        """
        f = StringIO(response_content)
        reader = csv.DictReader(f)

        principles = []
        for row in reader:
            principle = Principle(
                book_title=book_title,
                name=row.get("Name", "").strip(),
                details=row.get("Details", "").strip()
            )

            # Add the principle to the session before establishing relationships
            self.session.add(principle)

            # Associate tags
            tags = [tag.strip() for tag in row.get("Tags", "").split(",") if tag]
            for tag_name in tags:
                tag = self._get_or_create_tag(tag_name)
                principle.tags.append(tag)

            # Associate use cases
            use_cases = [use_case.strip() for use_case in row.get("Use Cases", "").split(",") if use_case]
            for use_case_name in use_cases:
                use_case = self._get_or_create_use_case(use_case_name)
                principle.use_cases.append(use_case)

            principles.append(principle)

        self.session.commit()
        return principles


    def get_principles_by_id(self, ids: List[int]) -> List[Principle]:
        """
        Retrieves principles by their IDs.

        Args:
            ids (List[int]): A list of principle IDs to retrieve.

        Returns:
            List[Principle]: A list of matching Principle instances.
        """
        return self.session.query(Principle).filter(Principle.id.in_(ids)).all()

    def get_principles_by_use_case_and_tags(
        self, use_cases: List[str], tags: List[str], union: bool = True
    ) -> List[Principle]:
        """
        Retrieves principles that match the given use cases and tags.

        Args:
            use_cases (List[str]): A list of use cases to filter by.
            tags (List[str]): A list of tags to filter by.
            union (bool): If True, returns principles that match any use case or tag (OR logic).
                          If False, returns principles that match both the use cases and tags (AND logic).

        Returns:
            List[Principle]: A list of matching Principle instances.
        """
        query = self.session.query(Principle)

        if union:
            query = query.join(Principle.tags).join(Principle.use_cases).filter(
                (Tag.name.in_(tags)) | (UseCase.name.in_(use_cases))
            )
        else:
            query = query.join(Principle.tags).join(Principle.use_cases).filter(
                Tag.name.in_(tags),
                UseCase.name.in_(use_cases)
            )

        return query.all()

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
