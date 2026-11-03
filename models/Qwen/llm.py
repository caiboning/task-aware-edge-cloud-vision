
import json
import re
from pathlib import Path

from llama_cpp import Llama


# ==============================
# Model Configuration
# ==============================

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "Qwen3-0.6B-Q4_0.gguf"

N_CTX = 2048
N_THREADS = 4
MAX_TOKENS = 384


# ==============================
# Supported Task Types
# ==============================

TASK_TYPES = [
    "existence",
    "counting",
    "listing",
    "attribute",
    "ocr",
    "complex"
]


# ==============================
# Semantic Classification Prompt
# ==============================

TASK_SYSTEM_PROMPT = """
You are a semantic intent classifier for an
edge-cloud visual question answering system.

You receive only the user's question.
You cannot see the image.

Your job is to understand what the user wants
to know about the image.

Do not answer the question.
Do not guess what is in the image.
Do not invent detection results.

Classify each request into one of six types.

TASK TYPES:

existence:
The user wants to know whether a particular
object or object category is present.

counting:
The user wants to know the number of objects
or instances of an object category.

listing:
The user wants to know which objects or
object categories are present.

attribute:
The user asks about visible properties,
including color, appearance, shape, clothing,
or other visual characteristics. This category
is about what an object looks like, not what it
is doing.

ocr:
The user wants to read, transcribe, or extract
written text from the image.
This includes words on signs, paper, screens,
books, labels, and notebooks. Classify by intent:
if the user asks what written words say or asks
to read/transcribe them, use ocr even if the
question names a sign or label as its target.
Examples: "What does the sign say?",
"What is written on this label?", and
"Read the words on the screen."

complex:
The user asks about actions, activities,
spatial relationships, scene conditions,
weather visible in a scene, interactions,
visual reasoning involving context, or predictions
about what could happen next. Use this category
for any request that does not fit the other five
categories. Complex tasks are sent to the cloud
model for answering.

IMPORTANT DISTINCTIONS:

- Checking whether an object exists is existence.
- Asking how many objects exist is counting.
- Asking which objects exist is listing.
- Reading written words is ocr, not listing.
- Asking what a sign, label, page, or screen says
  is ocr, not attribute.
- Asking about colors is attribute.
- Asking what a person or animal is doing is
  complex, not attribute.
- Understanding scene conditions is complex.
- Predicting future events or hypothetical outcomes
  is complex.
- Classify requests by their meaning, not exact
  wording or grammar.
- If no other task type clearly applies, use complex.

CLASSIFICATION EXAMPLES:

Question: "What does the sign say?"
Task type: ocr
Target: image
Attribute: text

Question: "What color is the car?"
Task type: attribute
Target: car
Attribute: color

Question: "What is the person doing?"
Task type: complex
Target: person
Attribute: action

Question: "What could happen next?"
Task type: complex
Target: image
Attribute: future outcome

TARGET:

Extract the main object or subject of the request.

Use a short noun or noun phrase.
Use singular object category names when possible.
Use "image" for global scene questions.
Use null if the target is unknown.

ATTRIBUTE:

For attribute tasks, describe the requested
property, such as "color" or "clothing".

For complex tasks, briefly identify the
requested aspect, such as "action",
"spatial_relationship", or "weather".

For ocr tasks, use "text".

For existence, counting, and listing tasks,
use JSON null unless a specific property
is necessary.

For complex tasks, briefly describe the
requested reasoning or visual context.

MULTIPLE REQUESTS:

If the user asks multiple independent questions,
return one task for each request.

OUTPUT RULES:

Return one JSON object containing a "tasks" array.

Each task must contain:
- task_type
- target
- attribute

Return only valid JSON.
Do not include explanations.
Do not copy the full user question.
Do not use the string "null".
Use JSON null instead.

/no_think
"""


# ==============================
# Structured Output Schema
# ==============================

TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": TASK_TYPES
                    },
                    "target": {
                        "type": ["string", "null"]
                    },
                    "attribute": {
                        "type": ["string", "null"]
                    }
                },
                "required": [
                    "task_type",
                    "target",
                    "attribute"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["tasks"],
    "additionalProperties": False
}


# ==============================
# Local LLM
# ==============================

class LocalLLM:

    def __init__(self):

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        print("Loading Qwen3-0.6B...")

        self.model = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=0,
            verbose=False
        )

        print("Qwen3-0.6B loaded successfully.")

    # ==============================
    # General Question Answering
    # ==============================

    def ask(self, question: str) -> str:

        response = self.model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Answer clearly and concisely. "
                        "/no_think"
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=256,
            temperature=0.1
        )

        answer = (
            response["choices"][0]["message"]
            .get("content") or ""
        )

        return self._clean_response(answer)

    # ==============================
    # Semantic Task Classification
    # ==============================

    def understand_task(self, question: str) -> dict:

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        response = self.model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": TASK_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            response_format={
                "type": "json_object",
                "schema": TASK_SCHEMA
            }
        )

        raw_output = (
            response["choices"][0]["message"]
            .get("content") or ""
        )

        cleaned = self._clean_response(raw_output)

        if not cleaned:
            raise ValueError(
                "LLM returned an empty response."
            )

        try:
            plan = json.loads(cleaned)

        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON from LLM: {cleaned}"
            ) from error

        plan = self._normalize_plan(plan)

        self._validate_plan(plan)

        return plan

    # ==============================
    # Response Cleaning
    # ==============================

    @staticmethod
    def _clean_response(text: str) -> str:

        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL
        )

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text.strip(),
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"\s*```$",
            "",
            text.strip()
        )

        return text.strip()

    # ==============================
    # Plan Normalization
    # ==============================

    @staticmethod
    def _normalize_plan(plan: dict) -> dict:

        if not isinstance(plan, dict):
            raise ValueError(
                "Task plan must be a JSON object."
            )

        tasks = plan.get("tasks")

        if not isinstance(tasks, list):
            raise ValueError(
                "Tasks must be a list."
            )

        for task in tasks:

            if not isinstance(task, dict):
                raise ValueError(
                    "Each task must be an object."
                )

            task_type = task.get("task_type")

            if isinstance(task_type, str):
                task_type = task_type.strip().lower()

            task["task_type"] = (
                task_type
                if task_type in TASK_TYPES
                else "complex"
            )

            for field in ("target", "attribute"):

                value = task.get(field)

                if isinstance(value, str):

                    value = value.strip()

                    if value.lower() in (
                        "null",
                        "none",
                        ""
                    ):
                        task[field] = None

                    else:
                        task[field] = value

        return plan

    # ==============================
    # Plan Validation
    # ==============================

    @staticmethod
    def _validate_plan(plan: dict):

        if set(plan.keys()) != {"tasks"}:
            raise ValueError(
                "Invalid top-level fields."
            )

        tasks = plan["tasks"]

        if not tasks:
            raise ValueError(
                "No tasks generated."
            )

        required_fields = {
            "task_type",
            "target",
            "attribute"
        }

        for index, task in enumerate(tasks):

            if not isinstance(task, dict):
                raise ValueError(
                    f"Task {index} must be an object."
                )

            if set(task.keys()) != required_fields:
                raise ValueError(
                    f"Task {index}: invalid fields."
                )

            if task["task_type"] not in TASK_TYPES:
                raise ValueError(
                    f"Task {index}: invalid task type."
                )

            for field in ("target", "attribute"):

                value = task[field]

                if value is not None and not isinstance(
                    value, str
                ):
                    raise ValueError(
                        f"Task {index}: invalid {field}."
                    )


# ==============================
# Model Loading Test
# ==============================

if __name__ == "__main__":
    llm = LocalLLM()
