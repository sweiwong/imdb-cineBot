from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise ImportError(
        "python-dotenv is required. Install it with `pip install python-dotenv`."
    ) from exc


class GraphState(TypedDict, total=False):
    query: str
    chat_history: List[Tuple[str, str]]
    constraints: Dict[str, Any]
    preference_profile: Dict[str, Any]
    preference_summary: str
    constraint_summary: str
    intent: str
    status: str
    answer: str
    docs: List[Document]
    result_rows: List[Dict[str, Any]]
    result_count: int
    displayed_count: int
    ranking_details: Dict[str, Any]
    used_constraint_fallback: bool
    assistant_mode: str
    formatted_response: str
    error: str


class IMDbLangGraphApp:
    """LangGraph version of the IMDb content discovery assistant."""

    CATALOG_RESULT_LIMIT = 20

    def __init__(self, dataset_path: Optional[str] = None) -> None:
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

        self.dataset_path = self._resolve_dataset_path(dataset_path)
        self.clean_df = self._prepare_dataframe(pd.read_csv(self.dataset_path))
        self.movie_docs = [self._row_to_document(row) for _, row in self.clean_df.iterrows()]

        self.known_certificates = [
            "G",
            "PG",
            "PG-13",
            "R",
            "NC-17",
            "Not Rated",
            "Approved",
            "TV-14",
            "TV-PG",
            "TV-MA",
        ]
        self.known_genres = sorted(
            self.clean_df["Genre"].dropna().astype(str).unique().tolist(),
            key=len,
            reverse=True,
        )
        self.movie_hint_terms = {
            "movie",
            "movies",
            "film",
            "films",
            "watch",
            "recommend",
            "recommendation",
            "similar",
            "genre",
            "actor",
            "actress",
            "cast",
            "director",
            "rating",
            "imdb",
            "runtime",
            "duration",
            "certificate",
            "plot",
            "story",
            "trailer",
            "cinema",
            "mood",
            "family",
        }

        self.title_lookup = set(self.clean_df["Title"].astype(str).map(self._normalize_lookup_text).tolist())
        self.director_lookup = set(self.clean_df["Director"].astype(str).map(self._normalize_lookup_text).tolist())

        self.embedding_model: Optional[OpenAIEmbeddings] = None
        self.vector_store: Optional[FAISS] = None
        self.llm: Optional[ChatOpenAI] = None
        self.prompt = self._build_prompt()
        self.graph = self._build_graph().compile()

    def invoke(self, query: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        initial_state: GraphState = {
            "query": query,
            "chat_history": chat_history or [],
        }
        return dict(self.graph.invoke(initial_state))

    def run(self, query: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> str:
        state = self.invoke(query, chat_history=chat_history)
        return state.get("formatted_response", state.get("answer", "No response generated."))

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(GraphState)
        graph.add_node("prepare", self._prepare_request)
        graph.add_node("search_agent", self._search_agent)
        graph.add_node("recommendation_agent", self._recommendation_agent)
        graph.add_node("catalog_agent", self._catalog_agent)
        graph.add_node("clarification_agent", self._clarification_agent)
        graph.add_node("finalize", self._finalize_response)

        graph.set_entry_point("prepare")
        graph.add_conditional_edges(
            "prepare",
            self._route_after_prepare,
            {
                "search_agent": "search_agent",
                "recommendation_agent": "recommendation_agent",
                "catalog_agent": "catalog_agent",
                "clarification_agent": "clarification_agent",
                "finalize": "finalize",
            },
        )
        graph.add_edge("search_agent", "finalize")
        graph.add_edge("recommendation_agent", "finalize")
        graph.add_edge("catalog_agent", "finalize")
        graph.add_edge("clarification_agent", "finalize")
        graph.add_edge("finalize", END)
        return graph

    def _prepare_request(self, state: GraphState) -> GraphState:
        query = str(state.get("query", "")).strip()
        chat_history = state.get("chat_history", [])

        if not query:
            return {
                "query": query,
                "chat_history": chat_history,
                "status": "invalid_input",
                "answer": "Tell me what kind of movie experience you want and I will narrow it down.",
                "assistant_mode": "guardrail",
            }

        if not self._is_probably_movie_related(query):
            return {
                "query": query,
                "chat_history": chat_history,
                "status": "off_topic",
                "answer": "I am focused on movie discovery. Ask me about a title, genre, director, actor, runtime, rating target, or the kind of vibe you want.",
                "assistant_mode": "guardrail",
            }

        constraints = self._extract_query_constraints(query)
        preference_profile = self._extract_preference_profile(query, chat_history)
        intent = self._detect_intent(query, constraints)

        if len(query) < 3:
            intent = "clarification"

        return {
            "query": query,
            "chat_history": chat_history,
            "constraints": constraints,
            "constraint_summary": self._summarize_constraints(constraints),
            "preference_profile": preference_profile,
            "preference_summary": self._summarize_preferences(preference_profile),
            "intent": intent,
        }

    def _route_after_prepare(self, state: GraphState) -> str:
        if state.get("status") in {"invalid_input", "off_topic", "error"}:
            return "finalize"

        intent = state.get("intent")
        if intent == "catalog":
            return "catalog_agent"
        if intent == "recommendation":
            return "recommendation_agent"
        if intent == "clarification":
            return "clarification_agent"
        return "search_agent"

    def _search_agent(self, state: GraphState) -> GraphState:
        return self._run_retrieval_agent(state, assistant_mode="search")

    def _recommendation_agent(self, state: GraphState) -> GraphState:
        return self._run_retrieval_agent(state, assistant_mode="recommendation")

    def _catalog_agent(self, state: GraphState) -> GraphState:
        constraints = state.get("constraints", {})

        if not self._has_hard_constraints(constraints):
            return {
                "status": "needs_clarification",
                "answer": "I can list matching titles, but I need at least one concrete filter such as genre, rating, year, certificate, or runtime.",
                "docs": [],
                "assistant_mode": "catalog",
                "constraints": constraints,
                "constraint_summary": self._summarize_constraints(constraints),
            }

        matched_df = self._filter_catalog_matches(self.clean_df, constraints)
        if matched_df.empty:
            return {
                "status": "no_match",
                "answer": f"I could not find any titles matching {self._summarize_constraints(constraints)} in this dataset.",
                "docs": [],
                "assistant_mode": "catalog",
                "constraints": constraints,
                "constraint_summary": self._summarize_constraints(constraints),
            }

        displayed_df = matched_df.head(self.CATALOG_RESULT_LIMIT).copy()
        displayed_docs = [self._row_to_document(row) for _, row in displayed_df.iterrows()]
        ranking_details = {
            doc.metadata.get("movie_id"): {
                "final_score": None,
                "semantic_score": None,
                "reason_tags": self._build_reason_tags(doc, constraints, preference_profile={}),
            }
            for doc in displayed_docs
        }

        result_count = len(matched_df)
        displayed_count = len(displayed_df)
        if result_count > displayed_count:
            answer = (
                f"I found {result_count} matching titles for {self._summarize_constraints(constraints)}. "
                f"I am showing the top {displayed_count}, sorted by IMDb rating and recency."
            )
        else:
            answer = f"I found {result_count} matching titles for {self._summarize_constraints(constraints)}."

        return {
            "status": "ok",
            "answer": answer,
            "docs": displayed_docs,
            "result_rows": displayed_df.to_dict("records"),
            "result_count": result_count,
            "displayed_count": displayed_count,
            "ranking_details": ranking_details,
            "assistant_mode": "catalog",
            "used_constraint_fallback": False,
            "constraints": constraints,
            "constraint_summary": self._summarize_constraints(constraints),
            "preference_summary": "catalog list mode uses explicit filters rather than soft taste signals",
        }

    def _clarification_agent(self, state: GraphState) -> GraphState:
        return {
            "status": "needs_clarification",
            "answer": "Give me one signal to work with such as genre, a movie you liked, an actor, mood, rating floor, or runtime.",
            "docs": [],
            "assistant_mode": "clarification",
            "constraints": state.get("constraints", {}),
            "constraint_summary": state.get("constraint_summary", "none"),
        }

    def _finalize_response(self, state: GraphState) -> GraphState:
        status = state.get("status", "unknown")
        if status != "ok":
            return {"formatted_response": state.get("answer", "Something went wrong.")}

        assistant_mode = state.get("assistant_mode", "search")
        constraint_line = f"**Constraint read:** {state.get('constraint_summary', 'none')}"

        if assistant_mode == "catalog":
            result_count = state.get("result_count", 0)
            displayed_count = state.get("displayed_count", 0)
            count_line = f"**Catalog match count:** {result_count}"
            shown_line = f"**Showing:** {displayed_count} title(s)"
            catalog_section = self._format_catalog_matches(state.get("result_rows", []))
            formatted = (
                f"{state['answer']}\n\n"
                f"{count_line}\n{shown_line}\n{constraint_line}\n\n"
                f"### Matching Movies\n{catalog_section}"
            )
            return {"formatted_response": formatted}

        movie_section = self._format_movie_matches(
            state.get("docs", []),
            ranking_details=state.get("ranking_details", {}),
        )
        preference_line = f"**Preference read:** {state.get('preference_summary', 'none')}"
        fallback_line = ""
        if state.get("used_constraint_fallback"):
            fallback_line = "\n**Fallback note:** No exact hard-constraint matches were found, so the assistant returned the closest high-similarity titles instead."

        compliance_line = ""
        constraints = state.get("constraints", {})
        docs = state.get("docs", [])
        compliance_report = self._evaluate_constraint_compliance(docs, constraints)
        if compliance_report.get("has_hard_constraints"):
            compliance_line = f"\n**Constraint compliance:** {compliance_report.get('compliance_rate', 0.0):.0%}"

        formatted = (
            f"{state['answer']}\n\n"
            f"{preference_line}\n{constraint_line}{compliance_line}{fallback_line}\n\n"
            f"### Top Matched Movies\n{movie_section}"
        )
        return {"formatted_response": formatted}

    def _run_retrieval_agent(self, state: GraphState, assistant_mode: Literal["search", "recommendation"]) -> GraphState:
        try:
            self._ensure_runtime_components()
            query = state["query"]
            chat_history = state.get("chat_history", [])
            constraints = state.get("constraints", {})
            preference_profile = state.get("preference_profile", {})
            retrieval_query = self._rewrite_query_for_retrieval(query, assistant_mode, preference_profile)
            candidates_with_scores = self._retrieve_docs_with_scores(retrieval_query, k=30)
            reranked = self._rerank_with_constraints(candidates_with_scores, constraints, preference_profile, top_k=5)
            docs = reranked["docs"]

            if not docs:
                return {
                    "status": "no_match",
                    "answer": "I could not find a confident movie match yet. Add one extra signal such as genre, a reference title, rating floor, or runtime.",
                    "docs": [],
                    "assistant_mode": assistant_mode,
                    "constraints": constraints,
                    "constraint_summary": reranked["constraint_summary"],
                    "preference_summary": reranked["preference_summary"],
                }

            answer = self._generate_answer(query, docs, chat_history, reranked["preference_summary"])
            return {
                "status": "ok",
                "answer": answer,
                "docs": docs,
                "constraints": reranked["constraints"],
                "constraint_summary": reranked["constraint_summary"],
                "preference_profile": reranked["preference_profile"],
                "preference_summary": reranked["preference_summary"],
                "ranking_details": reranked["ranking_details"],
                "used_constraint_fallback": reranked["used_constraint_fallback"],
                "assistant_mode": assistant_mode,
            }
        except Exception as exc:
            return {
                "status": "error",
                "answer": f"Something broke while processing the query: {exc}",
                "docs": [],
                "assistant_mode": assistant_mode,
                "error": str(exc),
            }

    def _ensure_runtime_components(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in .env before using the LangGraph app.")

        if self.embedding_model is None:
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        if self.vector_store is None:
            splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=40)
            chunked_docs = splitter.split_documents(self.movie_docs)
            self.vector_store = FAISS.from_documents(chunked_docs, self.embedding_model)
        if self.llm is None:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an IMDb movie discovery concierge. Your job is to help users discover movies with taste, clarity, and strong reasoning. "
                    "Use only retrieved context for factual details. Never invent titles, ratings, or credits. "
                    "Sound like a sharp, enthusiastic movie expert: concise, engaging, and insightful rather than generic. "
                    "When recommending, explain why the picks fit by referencing genre, tone, pacing, director, era, runtime, or ratings when available. "
                    "If the user gives weak input, ask one short clarifying question.",
                ),
                (
                    "human",
                    "Conversation history:\n{chat_history}\n\n"
                    "User question:\n{question}\n\n"
                    "Interpreted preference summary:\n{preference_summary}\n\n"
                    "Retrieved context:\n{context}\n\n"
                    "Return a concise markdown answer with this structure:\n"
                    "1) Quick Take\n2) Why These Picks Fit\n3) Recommended Movies\n4) One Smart Follow-Up\n",
                ),
            ]
        )

    def _generate_answer(
        self,
        question: str,
        docs: List[Document],
        chat_history: List[Tuple[str, str]],
        preference_summary: str,
    ) -> str:
        assert self.llm is not None
        prompt_value = self.prompt.format_prompt(
            chat_history=self._history_to_text(chat_history),
            question=question,
            preference_summary=preference_summary,
            context=self._docs_to_context(docs),
        )
        response = self.llm.invoke(prompt_value.to_messages())
        return response.content

    def _retrieve_docs_with_scores(self, query: str, k: int = 30):
        assert self.vector_store is not None
        return self.vector_store.similarity_search_with_score(query, k=k)

    def _rewrite_query_for_retrieval(self, query: str, assistant_mode: str, preference_profile: Dict[str, Any]) -> str:
        hints: List[str] = []
        if assistant_mode == "recommendation":
            hints.append("prioritize recommendation-worthy titles with strong fit and broad appeal")
        if preference_profile.get("preferred_genres"):
            hints.append("preferred genres: " + ", ".join(preference_profile["preferred_genres"][:4]))
        if preference_profile.get("mentioned_directors"):
            hints.append("reference directors: " + ", ".join(preference_profile["mentioned_directors"][:2]))
        if preference_profile.get("mentioned_titles"):
            hints.append("reference titles: " + ", ".join(preference_profile["mentioned_titles"][:2]))
        if not hints:
            return query
        return query + " | " + " | ".join(hints)

    def _extract_query_constraints(self, query: str) -> Dict[str, Any]:
        q = str(query).lower()
        constraints: Dict[str, Any] = {
            "genre": None,
            "certificate": None,
            "imdb_min": None,
            "imdb_max": None,
            "year_min": None,
            "year_max": None,
            "duration_min": None,
            "duration_max": None,
            "prefer_high_rating": False,
        }

        if any(phrase in q for phrase in ["highly rated", "top rated", "best rated", "highest rated"]):
            constraints["prefer_high_rating"] = True

        for genre in self.known_genres:
            if re.search(rf"\b{re.escape(genre.lower())}\b", q):
                constraints["genre"] = genre
                break

        for cert in sorted(self.known_certificates, key=len, reverse=True):
            if re.search(rf"\b{re.escape(cert.lower())}\b", q):
                constraints["certificate"] = cert
                break

        rating_min_patterns = [
            r"(?:imdb\s*rating|rating|rated)\s*(?:above|over|greater than|at least|>=)\s*(\d+(?:\.\d+)?)",
            r"(?:above|over|greater than|at least|>=)\s*(\d+(?:\.\d+)?)\s*(?:imdb\s*rating|rating)",
            r"(?:minimum|min)\s*(?:imdb\s*rating|rating)\s*(\d+(?:\.\d+)?)",
            r"(?:rated)\s*(\d+(?:\.\d+)?)\s*(?:and above|or above|plus)",
            r"(\d+(?:\.\d+)?)\s*(?:and above|or above)\s*(?:on imdb|imdb|rating)?",
        ]
        rating_max_patterns = [
            r"(?:imdb\s*rating|rating|rated)\s*(?:below|under|less than|at most|<=)\s*(\d+(?:\.\d+)?)",
            r"(?:below|under|less than|at most|<=)\s*(\d+(?:\.\d+)?)\s*(?:imdb\s*rating|rating)",
            r"(?:maximum|max)\s*(?:imdb\s*rating|rating)\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in rating_min_patterns:
            match = re.search(pattern, q)
            if match:
                constraints["imdb_min"] = float(match.group(1))
                constraints["prefer_high_rating"] = True
                break

        for pattern in rating_max_patterns:
            match = re.search(pattern, q)
            if match:
                constraints["imdb_max"] = float(match.group(1))
                break

        match = re.search(r"(?:after|newer than)\s*(19\d{2}|20\d{2})", q)
        if match:
            constraints["year_min"] = int(match.group(1)) + 1
        match = re.search(r"(?:since|from|after or in)\s*(19\d{2}|20\d{2})", q)
        if match and constraints["year_min"] is None:
            constraints["year_min"] = int(match.group(1))
        match = re.search(r"(?:before|older than)\s*(19\d{2}|20\d{2})", q)
        if match:
            constraints["year_max"] = int(match.group(1)) - 1

        duration_min_match = re.search(
            r"(?:over|above|more than|at least|>=)\s*(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|minutes?|mins?|min)",
            q,
        )
        if duration_min_match:
            constraints["duration_min"] = self._duration_to_minutes(
                float(duration_min_match.group(1)), duration_min_match.group(2)
            )

        duration_max_match = re.search(
            r"(?:under|below|less than|at most|<=)\s*(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|minutes?|mins?|min)",
            q,
        )
        if duration_max_match:
            constraints["duration_max"] = self._duration_to_minutes(
                float(duration_max_match.group(1)), duration_max_match.group(2)
            )

        return constraints

    def _extract_preference_profile(
        self,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []

        history_text = " ".join(user_msg for user_msg, _ in chat_history[-4:])
        combined_text = f"{query} {history_text}"
        combined_norm = self._normalize_lookup_text(combined_text)

        title_rows = self.clean_df[["Title", "Year", "Genre"]].to_dict("records")
        director_rows = self.clean_df[["Director", "Genre"]].to_dict("records")

        mentioned_titles: List[str] = []
        for row in title_rows:
            title_norm = self._normalize_lookup_text(row["Title"])
            if title_norm and re.search(rf"\b{re.escape(title_norm)}\b", combined_norm):
                mentioned_titles.append(row["Title"])

        mentioned_directors: List[str] = []
        for row in director_rows:
            director_norm = self._normalize_lookup_text(row["Director"])
            if director_norm and re.search(rf"\b{re.escape(director_norm)}\b", combined_norm):
                mentioned_directors.append(row["Director"])

        preferred_genres: set[str] = set()
        for genre in self.known_genres:
            if re.search(rf"\b{re.escape(genre.lower())}\b", combined_text.lower()):
                preferred_genres.update(self._genre_tokens(genre))

        mood_map = {
            "dark": {"thriller", "crime", "mystery"},
            "light": {"comedy", "animation", "family"},
            "emotional": {"drama", "romance"},
            "mind bending": {"sci-fi", "mystery", "thriller"},
            "feel good": {"comedy", "family", "musical"},
            "intense": {"action", "thriller", "war"},
        }
        matched_moods: List[str] = []
        lowered = combined_text.lower()
        for phrase, genres in mood_map.items():
            if phrase in lowered:
                matched_moods.append(phrase)
                preferred_genres.update(genres)

        return {
            "mentioned_titles": sorted(set(mentioned_titles)),
            "mentioned_directors": sorted(set(mentioned_directors)),
            "preferred_genres": sorted(preferred_genres),
            "matched_moods": matched_moods,
        }

    def _detect_intent(self, query: str, constraints: Dict[str, Any]) -> str:
        q = query.lower()
        catalog_phrases = [
            "give me all",
            "show me all",
            "list all",
            "show all",
            "every documentary",
            "every movie",
            "every film",
        ]
        if any(phrase in q for phrase in catalog_phrases):
            return "catalog"
        if re.search(r"\b(all|every)\b", q) and self._has_hard_constraints(constraints):
            return "catalog"
        if any(x in q for x in ["recommend", "suggest", "similar to", "what should i watch", "something like", "in the mood"]):
            return "recommendation"
        return "search"

    def _is_probably_movie_related(self, query: str) -> bool:
        q_norm = self._normalize_lookup_text(query)
        q_tokens = set(q_norm.split())
        if not q_norm:
            return False
        if any(term in q_tokens for term in self.movie_hint_terms):
            return True
        if any(genre.lower() in q_norm for genre in self.known_genres):
            return True
        if any(cert.lower() in q_norm for cert in self.known_certificates):
            return True
        if re.search(r"\b(19\d{2}|20\d{2})\b", q_norm):
            return True
        if q_norm in self.title_lookup or q_norm in self.director_lookup:
            return True
        return False

    def _has_hard_constraints(self, constraints: Dict[str, Any]) -> bool:
        hard_keys = [
            "genre",
            "certificate",
            "imdb_min",
            "imdb_max",
            "year_min",
            "year_max",
            "duration_min",
            "duration_max",
        ]
        return any(constraints.get(key) is not None for key in hard_keys)

    def _evaluate_constraint_compliance(self, docs: List[Document], constraints: Dict[str, Any]) -> Dict[str, Any]:
        has_hard_constraints = self._has_hard_constraints(constraints)
        if not has_hard_constraints:
            return {"has_hard_constraints": False, "compliance_rate": None}
        if not docs:
            return {"has_hard_constraints": True, "compliance_rate": 0.0}
        compliant_count = sum(self._doc_satisfies_hard_constraints(doc, constraints) for doc in docs)
        return {
            "has_hard_constraints": True,
            "compliance_rate": compliant_count / len(docs),
        }

    def _doc_satisfies_hard_constraints(self, doc: Document, constraints: Dict[str, Any]) -> bool:
        metadata = doc.metadata
        if constraints.get("genre") and constraints["genre"].lower() not in str(metadata.get("genre", "")).lower():
            return False
        if constraints.get("certificate") and str(metadata.get("certificate", "")).lower() != constraints["certificate"].lower():
            return False

        imdb_rating = float(metadata.get("imdb_rating", 0.0))
        year = int(metadata.get("year", 0))
        duration = float(metadata.get("duration_minutes", 0.0))

        if constraints.get("imdb_min") is not None and imdb_rating < constraints["imdb_min"]:
            return False
        if constraints.get("imdb_max") is not None and imdb_rating > constraints["imdb_max"]:
            return False
        if constraints.get("year_min") is not None and year < constraints["year_min"]:
            return False
        if constraints.get("year_max") is not None and year > constraints["year_max"]:
            return False
        if constraints.get("duration_min") is not None and duration < constraints["duration_min"]:
            return False
        if constraints.get("duration_max") is not None and duration > constraints["duration_max"]:
            return False
        return True

    def _filter_catalog_matches(self, df: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
        filtered_df = df.copy()
        if constraints.get("genre"):
            filtered_df = filtered_df[
                filtered_df["Genre"].astype(str).str.contains(
                    re.escape(constraints["genre"]), case=False, regex=True, na=False
                )
            ]
        if constraints.get("certificate"):
            filtered_df = filtered_df[
                filtered_df["Certificates"].astype(str).str.lower() == constraints["certificate"].lower()
            ]
        if constraints.get("imdb_min") is not None:
            filtered_df = filtered_df[filtered_df["IMDb Rating"] >= constraints["imdb_min"]]
        if constraints.get("imdb_max") is not None:
            filtered_df = filtered_df[filtered_df["IMDb Rating"] <= constraints["imdb_max"]]
        if constraints.get("year_min") is not None:
            filtered_df = filtered_df[filtered_df["Year"] >= constraints["year_min"]]
        if constraints.get("year_max") is not None:
            filtered_df = filtered_df[filtered_df["Year"] <= constraints["year_max"]]
        if constraints.get("duration_min") is not None:
            filtered_df = filtered_df[filtered_df["Duration (minutes)"] >= constraints["duration_min"]]
        if constraints.get("duration_max") is not None:
            filtered_df = filtered_df[filtered_df["Duration (minutes)"] <= constraints["duration_max"]]
        return filtered_df.sort_values(["IMDb Rating", "Year", "Title"], ascending=[False, False, True]).reset_index(drop=True)

    def _rerank_with_constraints(
        self,
        candidates_with_scores,
        constraints: Dict[str, Any],
        preference_profile: Dict[str, Any],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        best_by_movie_id: Dict[str, Tuple[Document, float]] = {}
        for doc, distance in candidates_with_scores:
            movie_id = doc.metadata.get("movie_id")
            if movie_id not in best_by_movie_id or distance < best_by_movie_id[movie_id][1]:
                best_by_movie_id[movie_id] = (doc, float(distance))

        unique_candidates = list(best_by_movie_id.values())
        scored_all = []
        ranking_details: Dict[str, Any] = {}
        preferred_genres = set(preference_profile.get("preferred_genres", []))
        preferred_directors = set(preference_profile.get("mentioned_directors", []))

        for doc, distance in unique_candidates:
            metadata = doc.metadata
            semantic_score = self._semantic_similarity(distance)
            rating = float(metadata.get("imdb_rating", 0.0))
            doc_genres = self._genre_tokens(metadata.get("genre", ""))

            rating_bonus = (rating / 10.0) * 0.10 if constraints.get("prefer_high_rating") else 0.0
            genre_bonus = 0.08 if preferred_genres.intersection(doc_genres) else 0.0
            director_bonus = 0.10 if metadata.get("director") in preferred_directors else 0.0
            final_score = (0.72 * semantic_score) + rating_bonus + genre_bonus + director_bonus

            scored_all.append((final_score, doc, distance))
            ranking_details[metadata.get("movie_id")] = {
                "final_score": round(final_score, 4),
                "semantic_score": round(semantic_score, 4),
                "reason_tags": self._build_reason_tags(doc, constraints, preference_profile),
            }

        hard_match_scored = [item for item in scored_all if self._doc_satisfies_hard_constraints(item[1], constraints)]
        used_constraint_fallback = False
        selected_pool = hard_match_scored
        if self._has_hard_constraints(constraints) and not hard_match_scored:
            selected_pool = scored_all
            used_constraint_fallback = True

        selected_pool = sorted(selected_pool, key=lambda item: item[0], reverse=True)[:top_k]
        reranked_docs = [doc for _, doc, _ in selected_pool]
        return {
            "docs": reranked_docs,
            "used_constraint_fallback": used_constraint_fallback,
            "constraint_summary": self._summarize_constraints(constraints),
            "preference_summary": self._summarize_preferences(preference_profile),
            "ranking_details": ranking_details,
            "constraints": constraints,
            "preference_profile": preference_profile,
        }

    def _history_to_text(self, chat_history: List[Tuple[str, str]], max_turns: int = 6) -> str:
        if not chat_history:
            return "No prior conversation."
        lines: List[str] = []
        for user_message, bot_message in chat_history[-max_turns:]:
            lines.append(f"User: {user_message}")
            lines.append(f"Assistant: {bot_message}")
        return "\n".join(lines)

    def _docs_to_context(self, docs: List[Document], max_docs: int = 5) -> str:
        selected = docs[:max_docs]
        blocks: List[str] = []
        for idx, doc in enumerate(selected, start=1):
            metadata = doc.metadata
            blocks.append(
                f"[{idx}] {metadata.get('title')} ({metadata.get('year')}) | Genre: {metadata.get('genre')} | "
                f"IMDb: {metadata.get('imdb_rating')} | Director: {metadata.get('director')} | "
                f"Certificate: {metadata.get('certificate')} | Runtime: {metadata.get('duration_minutes')} min\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(blocks) if blocks else "No relevant context found."

    def _format_movie_matches(self, docs: List[Document], ranking_details: Optional[Dict[str, Any]] = None) -> str:
        lines: List[str] = []
        seen: set[str] = set()
        for doc in docs:
            metadata = doc.metadata
            movie_id = metadata.get("movie_id")
            if movie_id in seen:
                continue
            seen.add(movie_id)
            detail = (ranking_details or {}).get(movie_id, {})
            reason_tags = detail.get("reason_tags", [])
            why_fit = "; ".join(reason_tags) if reason_tags else "strong semantic match"
            lines.append(
                f"- **{metadata.get('title')} ({metadata.get('year')})** | Genre: {metadata.get('genre')} | "
                f"IMDb: {metadata.get('imdb_rating')} | MetaScore: {metadata.get('metascore')} | "
                f"Director: {metadata.get('director')} | Runtime: {metadata.get('duration_minutes')} min\n"
                f"  Why it fits: {why_fit}"
            )
        return "\n".join(lines) if lines else "- No confident matches found."

    def _format_catalog_matches(self, result_rows: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for row in result_rows:
            lines.append(
                f"- **{row.get('Title')} ({int(row.get('Year'))})** | Genre: {row.get('Genre')} | "
                f"IMDb: {row.get('IMDb Rating')} | Director: {row.get('Director')} | "
                f"Runtime: {row.get('Duration (minutes)')} min"
            )
        return "\n".join(lines) if lines else "- No catalog matches found."

    def _summarize_constraints(self, constraints: Dict[str, Any]) -> str:
        parts: List[str] = []
        if constraints.get("genre"):
            parts.append(f"genre={constraints['genre']}")
        if constraints.get("certificate"):
            parts.append(f"certificate={constraints['certificate']}")
        if constraints.get("imdb_min") is not None:
            parts.append(f"imdb>={constraints['imdb_min']}")
        if constraints.get("imdb_max") is not None:
            parts.append(f"imdb<={constraints['imdb_max']}")
        if constraints.get("year_min") is not None:
            parts.append(f"year>={constraints['year_min']}")
        if constraints.get("year_max") is not None:
            parts.append(f"year<={constraints['year_max']}")
        if constraints.get("duration_min") is not None:
            parts.append(f"duration>={int(constraints['duration_min'])}m")
        if constraints.get("duration_max") is not None:
            parts.append(f"duration<={int(constraints['duration_max'])}m")
        return ", ".join(parts) if parts else "none"

    def _summarize_preferences(self, preference_profile: Dict[str, Any]) -> str:
        parts: List[str] = []
        if preference_profile.get("mentioned_titles"):
            parts.append("reference titles=" + ", ".join(preference_profile["mentioned_titles"][:3]))
        if preference_profile.get("mentioned_directors"):
            parts.append("reference directors=" + ", ".join(preference_profile["mentioned_directors"][:2]))
        if preference_profile.get("preferred_genres"):
            parts.append("soft genres=" + ", ".join(preference_profile["preferred_genres"][:4]))
        if preference_profile.get("matched_moods"):
            parts.append("mood cues=" + ", ".join(preference_profile["matched_moods"][:2]))
        return "; ".join(parts) if parts else "no strong soft preferences detected"

    def _build_reason_tags(self, doc: Document, constraints: Dict[str, Any], preference_profile: Dict[str, Any]) -> List[str]:
        metadata = doc.metadata
        doc_genres = self._genre_tokens(metadata.get("genre", ""))
        tags: List[str] = []

        if constraints.get("genre") and constraints["genre"].lower() in str(metadata.get("genre", "")).lower():
            tags.append(f"matches requested genre ({constraints['genre']})")
        if constraints.get("certificate") and str(metadata.get("certificate", "")).lower() == constraints["certificate"].lower():
            tags.append(f"fits certificate ({constraints['certificate']})")
        if constraints.get("duration_max") is not None and float(metadata.get("duration_minutes", 0.0)) <= constraints["duration_max"]:
            tags.append("fits runtime target")
        if constraints.get("imdb_min") is not None and float(metadata.get("imdb_rating", 0.0)) >= constraints["imdb_min"]:
            tags.append("meets rating threshold")

        preferred_genres = set(preference_profile.get("preferred_genres", []))
        overlap = preferred_genres.intersection(doc_genres)
        if overlap:
            tags.append("shares genre cues: " + ", ".join(sorted(overlap)[:2]))
        if float(metadata.get("imdb_rating", 0.0)) >= 8.0:
            tags.append("strong IMDb signal")
        return tags[:3]

    def _semantic_similarity(self, distance: float) -> float:
        return 1.0 / (1.0 + max(float(distance), 0.0))

    def _row_to_document(self, row: pd.Series) -> Document:
        metadata = {
            "movie_id": row["movie_id"],
            "title": row["Title"],
            "year": int(row["Year"]),
            "genre": row["Genre"],
            "imdb_rating": float(row["IMDb Rating"]),
            "metascore": float(row["MetaScore"]),
            "director": row["Director"],
            "star_cast": row["Star Cast"],
            "certificate": row["Certificates"],
            "duration_minutes": float(row["Duration (minutes)"]),
            "poster_src": row["Poster-src"],
        }
        return Document(page_content=row["search_text"], metadata=metadata)

    def _prepare_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        clean_df = raw_df.drop_duplicates().reset_index(drop=True).copy()
        text_cols = ["Title", "Genre", "Director", "Star Cast", "Certificates"]
        for col in text_cols:
            clean_df[col] = clean_df[col].apply(lambda value: re.sub(r"\s+", " ", str(value)).strip())

        def build_base_movie_id(title: str, year: int) -> str:
            normalized_title = re.sub(r"[^a-z0-9]+", "_", str(title).lower()).strip("_")
            return f"{normalized_title}_{int(year)}"

        clean_df["movie_id_base"] = clean_df.apply(lambda row: build_base_movie_id(row["Title"], row["Year"]), axis=1)
        clean_df["_id_rank"] = clean_df.groupby("movie_id_base").cumcount() + 1
        clean_df["movie_id"] = np.where(
            clean_df["_id_rank"] == 1,
            clean_df["movie_id_base"],
            clean_df["movie_id_base"] + "_alt" + clean_df["_id_rank"].astype(str),
        )
        clean_df = clean_df.drop(columns=["movie_id_base", "_id_rank"])

        clean_df["search_text"] = clean_df.apply(
            lambda row: (
                f"Title: {row['Title']}. "
                f"Year: {int(row['Year'])}. "
                f"Genre: {row['Genre']}. "
                f"Director: {row['Director']}. "
                f"Star Cast: {row['Star Cast']}. "
                f"IMDb Rating: {row['IMDb Rating']}. "
                f"MetaScore: {row['MetaScore']}. "
                f"Certificate: {row['Certificates']}. "
                f"Duration (minutes): {row['Duration (minutes)']}."
            ),
            axis=1,
        )

        display_columns = [
            "movie_id",
            "Title",
            "Year",
            "Genre",
            "IMDb Rating",
            "MetaScore",
            "Director",
            "Star Cast",
            "Poster-src",
            "Duration (minutes)",
            "Certificates",
            "search_text",
        ]
        return clean_df[display_columns].copy()

    def _resolve_dataset_path(self, dataset_path: Optional[str]) -> Path:
        if dataset_path:
            path = Path(dataset_path)
            if path.exists():
                return path
        default_path = Path.cwd() / "imdb_dataset.csv"
        if default_path.exists():
            return default_path
        raise FileNotFoundError("Could not locate imdb_dataset.csv.")

    def _normalize_lookup_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).strip()

    def _genre_tokens(self, value: str) -> set[str]:
        return {part.strip().lower() for part in str(value).split(",") if part.strip()}

    def _duration_to_minutes(self, value: float, unit: str) -> float:
        unit = unit.lower()
        if unit.startswith("h"):
            return float(value) * 60.0
        return float(value)


if __name__ == "__main__":
    app = IMDbLangGraphApp()
    demo_queries = [
        "Recommend 3 highly rated action movies under 2 hours.",
        "I loved Dune and Arrival. Give me something smart, immersive, and recent.",
        "Give me all documentaries rated 8.0 and above.",
    ]

    for idx, query in enumerate(demo_queries, start=1):
        print(f"\n{'=' * 24} LangGraph Demo {idx} {'=' * 24}")
        print(app.run(query))