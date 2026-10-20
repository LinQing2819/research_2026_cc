"""Execute only audited pure definitions from the pinned official source.

AST extraction preserves the actual function bodies, without importing the
online LLM/curator stack or constructing mutable online agent state.
"""
from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from types import SimpleNamespace

from .common import TASK, read_json, require


def definitions(path, names, class_name=None, namespace=None):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    body = tree.body
    if class_name:
        body = next(x.body for x in body if isinstance(x, ast.ClassDef)
                    and x.name == class_name)
    selected = [x for x in body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                and x.name in names]
    require({x.name for x in selected} == set(names), f"Official definitions missing: {path}")
    env = {"math": math, "re": re, **(namespace or {})}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), env)
    return {name: env[name] for name in names}


class Official:
    def __init__(self, root):
        self.root = Path(root)
        path = self.root / "methods/expel_online_st.py"
        names = ["_build_fewshot_examples_block", "_build_solver_prompt",
                 "_cosine", "retrieve_memory"]
        funcs = definitions(path, names, "ExpelOnlineST")
        self.ref = SimpleNamespace()
        for name, func in funcs.items():
            setattr(self.ref, name, func.__get__(self.ref))
        self.task_template = read_json(self.root / "prompts/prompt_templates.json")[
            "mmlu_pro_engineering"]
        boxed = definitions(self.root / "tasks/math_answer_utils.py",
                            ["extract_last_boxed"])
        task_source = self.root / "tasks/task_mmlu_pro.py"
        # Use the official module's regex assignments and extractor unchanged.
        tree = ast.parse(task_source.read_text(encoding="utf-8"))
        assignments = [x for x in tree.body if isinstance(x, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "_ANSWER_RE_LIST"
                               for t in x.targets)]
        env = {"re": re, **boxed}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(task_source), "exec"), env)
        self.extract_letter = definitions(task_source, ["_extract_letter"], namespace=env)[
            "_extract_letter"]

    @staticmethod
    def public(row):
        require(isinstance(row["question"], str) and row["question"].strip(), "Empty question")
        opts = list(row["options"])
        require(2 <= len(opts) <= 10 and all(isinstance(x, str) for x in opts),
                "MMLU options must contain 2..10 strings")
        return dict(qid=str(row.get("question_id", row.get("qid", ""))),
                    question=row["question"], options=opts)

    @staticmethod
    def options(row):
        return "\n".join(f"{chr(65+i)}. {s}" for i, s in enumerate(row["options"]))

    def query(self, public):
        return f"{public['question']}\n{self.options(public)}"

    def problem(self, public):
        return self.task_template.format(question=public["question"], options=self.options(public))

    def cases(self, items):
        # Only fields actually rendered by the official solver are projected.
        projected = [{k: x[k] for k in ("question", "trajectory", "retrieval_score")}
                     for x in items]
        return self.ref._build_fewshot_examples_block(projected)

    def solver(self, problem, rules, items):
        projected = [{k: x[k] for k in ("question", "trajectory", "retrieval_score")}
                     for x in items]
        return self.ref._build_solver_prompt(problem, rules, projected)

    def reference_retrieval(self, pool, vectors, query_vector, query):
        """Real official retrieval, with fixed vectors and no seed demos."""
        ref = SimpleNamespace(
            name="offline_reference", top_k=3,
            _task_experience_pool={TASK: pool},
            _task_experience_vectors={TASK: vectors},
            _seed_experience_pool_from_manual_fewshots=lambda **kw: None,
            _encode=lambda texts: [query_vector],
            _cosine=self.ref._cosine,
        )
        return self.ref.retrieve_memory.__func__(ref, SimpleNamespace(name=TASK), query=query)
