"""
Import/Export API endpoints for questions (JSON, Moodle XML, QTI).
"""
import io
import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.database import QuestionCRUD

router = APIRouter(prefix="/api", tags=["import_export"])


def _escape_xml(text: str) -> str:
    """Escape special XML characters"""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# ============================================================================
# IMPORT APIs
# ============================================================================

@router.post("/import/json")
async def import_json(file: UploadFile, db: Session = Depends(get_db)):
    """Import questions from JSON file"""
    content = await file.read()
    try:
        data = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="File JSON không hợp lệ")

    questions_data = data if isinstance(data, list) else data.get("questions", [])
    if not questions_data:
        raise HTTPException(status_code=400, detail="Không tìm thấy câu hỏi trong file")

    imported = []
    for q in questions_data:
        question = QuestionCRUD.create(
            db,
            content=q.get("content", q.get("question", "")),
            options=json.dumps(q.get("options", [])) if q.get("options") else None,
            answer=q.get("answer", ""),
            explanation=q.get("explanation", ""),
            subject=q.get("subject", "unknown"),
            topic=q.get("topic", ""),
            grade=q.get("grade", ""),
            question_type=q.get("question_type", q.get("type", "mcq")),
            difficulty=q.get("difficulty", "medium"),
            tags=json.dumps(q.get("tags", [])) if q.get("tags") else None,
            source="import_json",
        )
        imported.append(question.id)

    return {"ok": True, "count": len(imported), "message": f"Đã import {len(imported)} câu hỏi"}


@router.post("/import/moodle-xml")
async def import_moodle_xml(file: UploadFile, db: Session = Depends(get_db)):
    """Import questions from Moodle XML format"""
    import xml.etree.ElementTree as ET

    content = await file.read()
    try:
        root = ET.fromstring(content.decode("utf-8"))
    except ET.ParseError:
        raise HTTPException(status_code=400, detail="File XML không hợp lệ")

    imported = []
    for question_elem in root.findall(".//question"):
        qtype = question_elem.get("type", "multichoice")
        if qtype == "category":
            continue

        # Get question text
        name_elem = question_elem.find("name/text")
        questiontext_elem = question_elem.find("questiontext/text")

        content_text = ""
        if questiontext_elem is not None and questiontext_elem.text:
            content_text = questiontext_elem.text
        elif name_elem is not None and name_elem.text:
            content_text = name_elem.text

        if not content_text:
            continue

        # Get answers
        options = []
        correct_answer = ""
        for answer_elem in question_elem.findall("answer"):
            answer_text_elem = answer_elem.find("text")
            if answer_text_elem is not None and answer_text_elem.text:
                options.append(answer_text_elem.text)
                fraction = answer_elem.get("fraction", "0")
                if float(fraction) > 0:
                    correct_answer = answer_text_elem.text

        # Map Moodle question types
        question_type_map = {
            "multichoice": "mcq",
            "truefalse": "mcq",
            "shortanswer": "blank",
            "essay": "essay",
            "matching": "matching",
            "numerical": "blank",
        }

        question = QuestionCRUD.create(
            db,
            content=content_text,
            options=json.dumps(options) if options else None,
            answer=correct_answer,
            subject="imported",
            question_type=question_type_map.get(qtype, "mcq"),
            difficulty="medium",
            source="import_moodle",
        )
        imported.append(question.id)

    return {"ok": True, "count": len(imported), "message": f"Đã import {len(imported)} câu hỏi từ Moodle XML"}


@router.post("/import/qti")
async def import_qti(file: UploadFile, db: Session = Depends(get_db)):
    """Import questions from QTI (IMS Question & Test Interoperability) format"""
    import xml.etree.ElementTree as ET

    content = await file.read()
    try:
        root = ET.fromstring(content.decode("utf-8"))
    except ET.ParseError:
        raise HTTPException(status_code=400, detail="File QTI không hợp lệ")

    # Handle namespaces
    ns = {"qti": "http://www.imsglobal.org/xsd/imsqti_v2p1"}

    imported = []

    # Try QTI 2.1 format first
    for item in root.findall(".//qti:assessmentItem", ns) or root.findall(".//assessmentItem"):
        title = item.get("title", "")

        # Get item body
        item_body = item.find(".//qti:itemBody", ns) or item.find(".//itemBody")
        if item_body is None:
            continue

        content_text = "".join(item_body.itertext()).strip()
        if not content_text and title:
            content_text = title

        # Get choices
        options = []
        choice_interaction = item.find(".//qti:choiceInteraction", ns) or item.find(".//choiceInteraction")
        if choice_interaction is not None:
            for choice in choice_interaction.findall(".//qti:simpleChoice", ns) or choice_interaction.findall(".//simpleChoice"):
                choice_text = "".join(choice.itertext()).strip()
                if choice_text:
                    options.append(choice_text)

        # Get correct answer
        correct_answer = ""
        response_declaration = item.find(".//qti:responseDeclaration", ns) or item.find(".//responseDeclaration")
        if response_declaration is not None:
            correct_value = response_declaration.find(".//qti:value", ns) or response_declaration.find(".//value")
            if correct_value is not None and correct_value.text:
                identifier = correct_value.text
                for choice in (choice_interaction.findall(".//qti:simpleChoice", ns) if choice_interaction is not None else []):
                    if choice.get("identifier") == identifier:
                        correct_answer = "".join(choice.itertext()).strip()
                        break

        if content_text:
            question = QuestionCRUD.create(
                db,
                content=content_text,
                options=json.dumps(options) if options else None,
                answer=correct_answer,
                subject="imported",
                question_type="mcq",
                difficulty="medium",
                source="import_qti",
            )
            imported.append(question.id)

    return {"ok": True, "count": len(imported), "message": f"Đã import {len(imported)} câu hỏi từ QTI"}


# ============================================================================
# EXPORT APIs
# ============================================================================

@router.get("/export/json")
def export_questions_json(
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Export questions to JSON format"""
    questions = QuestionCRUD.get_all(db, limit=10000, subject=subject, topic=topic, difficulty=difficulty)

    data = []
    for q in questions:
        data.append({
            "content": q.content,
            "options": json.loads(q.options) if q.options else [],
            "answer": q.answer,
            "explanation": q.explanation,
            "subject": q.subject,
            "topic": q.topic,
            "grade": q.grade,
            "question_type": q.question_type,
            "difficulty": q.difficulty,
            "tags": json.loads(q.tags) if q.tags else [],
        })

    return StreamingResponse(
        io.BytesIO(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=questions.json"},
    )


@router.get("/export/moodle-xml")
def export_moodle_xml(
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Export questions to Moodle XML format"""
    questions = QuestionCRUD.get_all(db, limit=10000, subject=subject, topic=topic)

    xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<quiz>\n'

    for q in questions:
        qtype = "multichoice" if q.question_type == "mcq" else "shortanswer"

        xml_content += f'  <question type="{qtype}">\n'
        xml_content += f'    <name><text>{_escape_xml(q.content[:100])}</text></name>\n'
        xml_content += f'    <questiontext format="html"><text><![CDATA[{q.content}]]></text></questiontext>\n'

        if q.options:
            try:
                options = json.loads(q.options)
                for opt in options:
                    fraction = "100" if opt == q.answer else "0"
                    xml_content += f'    <answer fraction="{fraction}"><text><![CDATA[{opt}]]></text></answer>\n'
            except json.JSONDecodeError:
                pass

        if q.explanation:
            xml_content += f'    <generalfeedback format="html"><text><![CDATA[{q.explanation}]]></text></generalfeedback>\n'

        xml_content += '  </question>\n'

    xml_content += '</quiz>'

    return StreamingResponse(
        io.BytesIO(xml_content.encode("utf-8")),
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=questions_moodle.xml"},
    )


@router.get("/export/qti")
def export_qti(
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Export questions to QTI 2.1 format"""
    questions = QuestionCRUD.get_all(db, limit=10000, subject=subject, topic=topic)

    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<assessmentTest xmlns="http://www.imsglobal.org/xsd/imsqti_v2p1"
                xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:schemaLocation="http://www.imsglobal.org/xsd/imsqti_v2p1 http://www.imsglobal.org/xsd/qti/qtiv2p1/imsqti_v2p1.xsd"
                identifier="test_export"
                title="Exported Questions">
'''

    for i, q in enumerate(questions, 1):
        item_id = f"item_{i}"
        xml_content += f'''  <assessmentItem identifier="{item_id}" title="{_escape_xml(q.content[:50])}" adaptive="false" timeDependent="false">
    <responseDeclaration identifier="RESPONSE" cardinality="single" baseType="identifier">
'''

        if q.options and q.answer:
            try:
                options = json.loads(q.options)
                correct_idx = options.index(q.answer) if q.answer in options else 0
                xml_content += f'      <correctResponse><value>choice_{correct_idx}</value></correctResponse>\n'
            except (json.JSONDecodeError, ValueError):
                pass

        xml_content += '''    </responseDeclaration>
    <itemBody>
'''
        xml_content += f'      <p>{_escape_xml(q.content)}</p>\n'

        if q.options:
            try:
                options = json.loads(q.options)
                xml_content += '      <choiceInteraction responseIdentifier="RESPONSE" shuffle="false" maxChoices="1">\n'
                for j, opt in enumerate(options):
                    xml_content += f'        <simpleChoice identifier="choice_{j}">{_escape_xml(opt)}</simpleChoice>\n'
                xml_content += '      </choiceInteraction>\n'
            except json.JSONDecodeError:
                pass

        xml_content += '''    </itemBody>
  </assessmentItem>
'''

    xml_content += '</assessmentTest>'

    return StreamingResponse(
        io.BytesIO(xml_content.encode("utf-8")),
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=questions_qti.xml"},
    )
