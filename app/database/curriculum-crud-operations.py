"""
CRUD operations for Curriculum model.
"""

from typing import List, Optional

from sqlalchemy.orm import Session

from app.database.models import Curriculum


class CurriculumCRUD:
    @staticmethod
    def create(db: Session, **kwargs) -> Curriculum:
        curriculum = Curriculum(**kwargs)
        db.add(curriculum)
        db.commit()
        db.refresh(curriculum)
        return curriculum

    @staticmethod
    def bulk_create(db: Session, items: List[dict]) -> List[Curriculum]:
        db_items = [Curriculum(**item) for item in items]
        db.add_all(db_items)
        db.commit()
        for item in db_items:
            db.refresh(item)
        return db_items

    @staticmethod
    def get_all(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        subject: Optional[str] = None,
        grade: Optional[int] = None,
        chapter: Optional[str] = None,
    ) -> List[Curriculum]:
        query = db.query(Curriculum)

        if subject:
            query = query.filter(Curriculum.subject == subject)
        if grade:
            query = query.filter(Curriculum.grade == grade)
        if chapter:
            query = query.filter(Curriculum.chapter.contains(chapter))

        return query.order_by(Curriculum.grade, Curriculum.chapter).offset(skip).limit(limit).all()

    @staticmethod
    def get_by_subject_grade(db: Session, subject: str, grade: int) -> List[Curriculum]:
        return db.query(Curriculum).filter(
            Curriculum.subject == subject,
            Curriculum.grade == grade
        ).order_by(Curriculum.chapter).all()

    @staticmethod
    def get_topics(db: Session, subject: str, grade: int) -> List[str]:
        """Get unique topics for a subject and grade"""
        results = db.query(Curriculum.topic).filter(
            Curriculum.subject == subject,
            Curriculum.grade == grade,
            Curriculum.topic.isnot(None)
        ).distinct().all()
        return [r[0] for r in results if r[0]]

    @staticmethod
    def get_chapters(db: Session, subject: str, grade: int) -> List[str]:
        """Get unique chapters for a subject and grade"""
        results = db.query(Curriculum.chapter).filter(
            Curriculum.subject == subject,
            Curriculum.grade == grade,
            Curriculum.chapter.isnot(None)
        ).distinct().all()
        return [r[0] for r in results if r[0]]

    @staticmethod
    def search(db: Session, keyword: str, subject: str = None, grade: int = None) -> List[Curriculum]:
        """Search curriculum by keyword in topic, lesson, or knowledge"""
        query = db.query(Curriculum).filter(
            (Curriculum.topic.contains(keyword)) |
            (Curriculum.lesson.contains(keyword)) |
            (Curriculum.knowledge.contains(keyword))
        )
        if subject:
            query = query.filter(Curriculum.subject == subject)
        if grade:
            query = query.filter(Curriculum.grade == grade)
        return query.all()

    @staticmethod
    def delete_by_subject(db: Session, subject: str) -> int:
        count = db.query(Curriculum).filter(Curriculum.subject == subject).delete()
        db.commit()
        return count

    @staticmethod
    def delete_all(db: Session) -> int:
        count = db.query(Curriculum).delete()
        db.commit()
        return count

    @staticmethod
    def count(db: Session, subject: str = None, grade: int = None) -> int:
        query = db.query(Curriculum)
        if subject:
            query = query.filter(Curriculum.subject == subject)
        if grade:
            query = query.filter(Curriculum.grade == grade)
        return query.count()

    @staticmethod
    def get_stats(db: Session) -> dict:
        """Get statistics grouped by subject and grade"""
        from sqlalchemy import func

        stats = db.query(
            Curriculum.subject,
            Curriculum.grade,
            func.count(Curriculum.id).label("count")
        ).group_by(Curriculum.subject, Curriculum.grade).all()

        result = {}
        for subject, grade, count in stats:
            if subject not in result:
                result[subject] = {"total": 0, "grades": {}}
            result[subject]["grades"][grade] = count
            result[subject]["total"] += count

        return result
