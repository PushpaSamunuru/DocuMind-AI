from dataclasses import dataclass
from enum import Enum


class QueryIntent(str, Enum):
    GPA = "gpa"
    PROJECTS = "projects"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    UNIVERSITY = "university"
    DEGREE = "degree"
    SKILLS = "skills"
    CERTIFICATIONS = "certifications"
    EMAIL = "email"
    PHONE = "phone"
    DATE = "date"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    GENERAL = "general"


@dataclass(frozen=True)
class QueryRoute:
    intent: QueryIntent
    target_document: str | None = None
    requires_all_results: bool = False


class QueryRouter:
    @staticmethod
    def _detect_target_document(
        question: str,
    ) -> str | None:
        question_lower = question.lower()

        if any(
            phrase in question_lower
            for phrase in [
                "resume",
                "my cv",
                "the cv",
            ]
        ):
            return "resume"

        if any(
            phrase in question_lower
            for phrase in [
                "cover letter",
                "coverletter",
            ]
        ):
            return "cover"

        if any(
            phrase in question_lower
            for phrase in [
                "transcript",
                "grade report",
                "academic record",
            ]
        ):
            return "academic"

        return None

    def route(
        self,
        question: str,
    ) -> QueryRoute:
        question_lower = question.lower().strip()

        target_document = self._detect_target_document(
            question
        )

        comparison_words = [
            "compare",
            "comparison",
            "difference",
            "differences",
            "similarity",
            "similarities",
            "contrast",
            "versus",
            " vs ",
        ]

        if any(
            word in question_lower
            for word in comparison_words
        ):
            return QueryRoute(
                intent=QueryIntent.COMPARISON,
                target_document=target_document,
                requires_all_results=True,
            )

        if any(
            phrase in question_lower
            for phrase in [
                "summarize all",
                "summarise all",
                "summary of all",
                "summarize the documents",
                "summarise the documents",
                "summarize my resume",
                "summarise my resume",
                "resume summary",
                "document summary",
            ]
        ):
            return QueryRoute(
                intent=QueryIntent.SUMMARY,
                target_document=target_document,
                requires_all_results=True,
            )

        if any(
            word in question_lower
            for word in [
                "gpa",
                "grade point average",
            ]
        ):
            return QueryRoute(
                intent=QueryIntent.GPA,
                target_document=target_document,
            )

        if any(
            phrase in question_lower
            for phrase in [
                "project",
                "projects",
                "portfolio project",
                "portfolio projects",
            ]
        ):
            return QueryRoute(
                intent=QueryIntent.PROJECTS,
                target_document=target_document,
                requires_all_results=True,
            )

        if any(
            phrase in question_lower
            for phrase in [
                "work experience",
                "professional experience",
                "employment history",
                "job history",
                "what are experiences",
                "what experience",
                "experiences",
                "employment",
                "internship",
                "internships",
            ]
        ):
            return QueryRoute(
                intent=QueryIntent.EXPERIENCE,
                target_document=target_document,
                requires_all_results=True,
            )

        if any(
            word in question_lower
            for word in [
                "certification",
                "certifications",
                "certificate",
                "certificates",
                "certified",
            ]
        ):
            return QueryRoute(
                intent=QueryIntent.CERTIFICATIONS,
                target_document=target_document,
                requires_all_results=True,
            )

        if any(
            phrase in question_lower
            for phrase in [
                "technical skills",
                "technology skills",
                "programming skills",
                "what skills",
                "which skills",
                "list skills",
                "skills listed",
                "technologies",
                "programming languages",
                "libraries",
                "frameworks",
                "tools",
            ]
        ):
            return QueryRoute(
                intent=QueryIntent.SKILLS,
                target_document=target_document,
                requires_all_results=True,
            )

        has_university = any(
            word in question_lower
            for word in [
                "university",
                "college",
                "institution",
                "school",
            ]
        )

        has_degree = any(
            word in question_lower
            for word in [
                "degree",
                "master",
                "masters",
                "bachelor",
                "bachelors",
                "education qualification",
            ]
        )

        if has_university and has_degree:
            return QueryRoute(
                intent=QueryIntent.EDUCATION,
                target_document=target_document,
                requires_all_results=True,
            )

        if has_university:
            return QueryRoute(
                intent=QueryIntent.UNIVERSITY,
                target_document=target_document,
                requires_all_results=True,
            )

        if has_degree:
            return QueryRoute(
                intent=QueryIntent.DEGREE,
                target_document=target_document,
                requires_all_results=True,
            )

        if any(
            phrase in question_lower
            for phrase in [
                "education",
                "educational background",
                "academic background",
                "academic qualification",
                "academic qualifications",
            ]
        ):
            return QueryRoute(
                intent=QueryIntent.EDUCATION,
                target_document=target_document,
                requires_all_results=True,
            )

        if any(
            word in question_lower
            for word in [
                "email",
                "email address",
                "e-mail",
            ]
        ):
            return QueryRoute(
                intent=QueryIntent.EMAIL,
                target_document=target_document,
            )

        if any(
            phrase in question_lower
            for phrase in [
                "phone",
                "phone number",
                "mobile",
                "mobile number",
                "contact number",
                "telephone",
            ]
        ):
            return QueryRoute(
                intent=QueryIntent.PHONE,
                target_document=target_document,
            )

        if any(
            phrase in question_lower
            for phrase in [
                "date",
                "written date",
                "letter date",
                "dated",
                "when was",
                "when is",
            ]
        ):
            return QueryRoute(
                intent=QueryIntent.DATE,
                target_document=target_document,
            )

        return QueryRoute(
            intent=QueryIntent.GENERAL,
            target_document=target_document,
        )