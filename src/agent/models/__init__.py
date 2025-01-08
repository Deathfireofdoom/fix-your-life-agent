from sqlalchemy import Column, Integer, String, Text, DateTime, func, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Association tables for many-to-many relationships
principle_tags = Table(
    'principle_tags',
    Base.metadata,
    Column('principle_id', Integer, ForeignKey('principles.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)

principle_use_cases = Table(
    'principle_use_cases',
    Base.metadata,
    Column('principle_id', Integer, ForeignKey('principles.id'), primary_key=True),
    Column('use_case_id', Integer, ForeignKey('use_cases.id'), primary_key=True)
)

# Principle model
class Principle(Base):
    __tablename__ = 'principles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    book_title = Column(String(255), nullable=False)
    name = Column(Text, nullable=False)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    tags = relationship('Tag', secondary=principle_tags, back_populates='principles')
    use_cases = relationship('UseCase', secondary=principle_use_cases, back_populates='principles')

    def __repr__(self):
        return f"<Principle(id={self.id}, book_title='{self.book_title}', principle='{self.principle}')>"

# Tag model
class Tag(Base):
    __tablename__ = 'tags'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)

    # Relationships
    principles = relationship('Principle', secondary=principle_tags, back_populates='tags')

    def __repr__(self):
        return f"<Tag(id={self.id}, name='{self.name}')>"

# UseCase model
class UseCase(Base):
    __tablename__ = 'use_cases'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)

    # Relationships
    principles = relationship('Principle', secondary=principle_use_cases, back_populates='use_cases')

    def __repr__(self):
        return f"<UseCase(id={self.id}, name='{self.name}')>"
