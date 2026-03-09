"""
CRUD operations for Exam and ExamQuestion models.
"""

from typing import List, Optional

from sqlalchemy.orm import Session

from app.database.models import Exam, ExamQuestion, ExamVariant


class ExamCRUD:
    @staticmethod
    def create(db: Session, **kwargs) -> Exam:
        exam = Exam(**kwargs)
        db.add(exam)
        db.commit()
        db.refresh(exam)
        return exam

    @staticmethod
    def get(db: Session, exam_id: int) -> Optional[Exam]:
        return db.query(Exam).filter(Exam.id == exam_id).first()

    @staticmethod
    def get_all(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        subject: Optional[str] = None,
    ) -> List[Exam]:
        query = db.query(Exam)

        if subject:
            query = query.filter(Exam.subject == subject)

        return query.order_by(Exam.created_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def update(db: Session, exam_id: int, **kwargs) -> Optional[Exam]:
        exam = db.query(Exam).filter(Exam.id == exam_id).first()
        if exam:
            for key, value in kwargs.items():
                if hasattr(exam, key):
                    setattr(exam, key, value)
            db.commit()
            db.refresh(exam)
        return exam

    @staticmethod
    def delete(db: Session, exam_id: int) -> bool:
        exam = db.query(Exam).filter(Exam.id == exam_id).first()
        if exam:
            db.delete(exam)
            db.commit()
            return True
        return False

    @staticmethod
    def add_questions(db: Session, exam_id: int, question_ids: List[int]) -> Exam:
        exam = db.query(Exam).filter(Exam.id == exam_id).first()
        if not exam:
            return None

        # Get current max order
        max_order = db.query(ExamQuestion).filter(
            ExamQuestion.exam_id == exam_id
        ).count()

        for i, qid in enumerate(question_ids):
            eq = ExamQuestion(
                exam_id=exam_id,
                question_id=qid,
                order=max_order + i + 1
            )
            db.add(eq)

        exam.total_questions = max_order + len(question_ids)
        db.commit()
        db.refresh(exam)
        return exam

    @staticmethod
    def remove_question(db: Session, exam_id: int, question_id: int) -> bool:
        eq = db.query(ExamQuestion).filter(
            ExamQuestion.exam_id == exam_id,
            ExamQuestion.question_id == question_id
        ).first()

        if eq:
            db.delete(eq)
            # Update exam total
            exam = db.query(Exam).filter(Exam.id == exam_id).first()
            if exam:
                exam.total_questions = db.query(ExamQuestion).filter(
                    ExamQuestion.exam_id == exam_id
                ).count() - 1
            db.commit()
            return True
        return False

    @staticmethod
    def create_variant(db: Session, exam_id: int, variant_code: str, question_order: str, answer_mapping: str = None) -> ExamVariant:
        variant = ExamVariant(
            exam_id=exam_id,
            variant_code=variant_code,
            question_order=question_order,
            answer_mapping=answer_mapping
        )
        db.add(variant)
        db.commit()
        db.refresh(variant)
        return variant
