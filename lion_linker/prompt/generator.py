from __future__ import annotations

import copy
import json

from lion_linker.utils import load_prompt


class PromptGenerator:
    def __init__(self, prompt_file, few_shot_examples_file_path=None, tablellama_format=False):
        if prompt_file in {"base", "detailed", "few_shot", "tablellama"}:
            self.template = load_prompt(prompt_file)
        else:
            with open(prompt_file, "r") as file:
                self.template = file.read()

        self.tablellama_format = tablellama_format

        self.few_shot_examples = "N.A."
        if few_shot_examples_file_path is not None:
            with open(few_shot_examples_file_path, "r") as file:
                self.few_shot_examples = file.read()

    def _format_table(self, table: list[list[str]]) -> str:
        if self.tablellama_format:
            table_str = ""
            for row_idx, row in enumerate(table):
                if row_idx == 0:
                    table_str += "col: " + "| " + " | ".join(map(str, row)) + " |"
                else:
                    table_str += (
                        f" [SEP] row {row_idx}: " + "| " + " | ".join(map(str, row)) + " |"
                )
            return table_str
        else:
            return "\n".join(["|" + "|".join(map(str, row)) + "|" for row in table])

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        if value is None:
            return "N.A."
        return " ".join(str(value).split()) or "N.A."

    @staticmethod
    def _optimize_candidates(
        candidates: list[dict[str, str | list[dict[str, str]]]],
    ) -> list[dict[str, str | list[dict[str, str]]]]:
        optimized_candidates = []
        for candidate in candidates:
            optimized_candidate = {
                "id": candidate["id"],
                "name": candidate["name"],
                "description": candidate["description"],
                "types": [
                    {"id": t["id"], "name": t["name"]}
                    for t in candidate["types"]
                    if t["name"] is not None
                ],
            }
            optimized_candidates.append(optimized_candidate)
        return optimized_candidates

    def _format_candidates_text(
        self,
        candidates: list[dict[str, str | list[dict[str, str]]]],
        compact: bool = True,
        format_candidates: bool = True,
    ) -> str:
        optimized_candidates = self._optimize_candidates(candidates)
        if format_candidates:
            if not self.tablellama_format:
                return ",".join(
                    [
                        f"<id: {candidate['id']}; "
                        f"name: {candidate['name']}; "
                        f"description: {candidate['description'] if candidate['description'] is not None else 'N.A.'}; "  # noqa: E501
                        f"types: {','.join([t['name'] for t in candidate['types'] if t['name'] is not None])}>"  # noqa: E501
                        for candidate in optimized_candidates
                    ]
                )
            return ",".join(
                [
                    f"<{candidate['name']} "
                    f"[DESCRIPTION] {candidate['description'] if candidate['description'] is not None else 'None'} "  # noqa: E501
                    f"[TYPE] {','.join([t['name'] for t in candidate['types'] if t['name'] is not None])}>"  # noqa: E501
                    for candidate in optimized_candidates
                ]
            )
        if compact:
            lines = ["CANDIDATES (ID | TYPE | DESCRIPTION):"]
            if not optimized_candidates:
                lines.append("- N.A. | N.A. | N.A.")
                return "\n".join(lines)
            for candidate in optimized_candidates:
                candidate_id = self._normalize_text(candidate.get("id"))
                type_name = "N.A."
                for type_entry in candidate.get("types", []):
                    type_label = type_entry.get("name")
                    if type_label:
                        type_name = type_label
                        break
                description = self._normalize_text(
                    candidate.get("description") or candidate.get("name")
                )
                lines.append(f"- {candidate_id} | {type_name} | {description}")
            return "\n".join(lines)
        return json.dumps(optimized_candidates, separators=(",", ":"))

    def generate_prompt(
        self,
        table: list[list[str]],
        table_metadata: str | None,
        table_summary: str | None,
        column_name: str | None,
        entity_mention: str,
        candidates: list[dict[str, str | list[dict[str, str]]]],
        compact: bool = True,
        format_candidates: bool = True,
        answer_format: str | None = None,
    ):
        template = copy.deepcopy(self.template)

        formatted_table = self._format_table(table)
        table_metadata = self._normalize_text(table_metadata)
        table_summary = self._normalize_text(table_summary)
        if column_name is None:
            column_name = "N.A."
        candidates_text = self._format_candidates_text(
            candidates,
            compact=compact,
            format_candidates=format_candidates,
        )

        # Replace placeholders in the template with actual values
        # Define a dictionary with placeholders as keys and corresponding values
        replacements = {
            "[EXAMPLES]": self.few_shot_examples,
            "[TABLE]": formatted_table,
            "[TABLE METADATA]": table_metadata,
            "[SUMMARY]": table_summary,
            "[COLUMN NAME]": column_name,
            "[ENTITY MENTION]": entity_mention,
            "[CANDIDATES]": candidates_text,
            "[ANSWER_FORMAT]": answer_format,
        }

        # Replace each placeholder using the dictionary
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, str(value))

        return template

    def generate_multi_prompt(
        self,
        tasks: list[dict[str, object]],
        answer_format: str,
        compact: bool = True,
        format_candidates: bool = True,
    ) -> str:
        lines: list[str] = []
        lines.append("You perform entity linking over table cell mentions.")
        lines.append("For each task, rank the candidates and return a top list following the answer format.")
        lines.append("")
        lines.append("For each task you are given:")
        lines.append("  - The table header and the target row.")
        lines.append("  - The entity mention to link.")
        lines.append("  - The list of candidate entities.")
        lines.append("")
        if format_candidates:
            if self.tablellama_format:
                lines.append(
                    "Candidates are given as: <name [DESCRIPTION] ... [TYPE] ...>"
                )
            else:
                lines.append(
                    "Candidates are given as: <id: ...; name: ...; description: ...; types: ...>"
                )
            lines.append("types is a short category such as film, television film, novel, album, etc.")
        elif compact:
            lines.append("Candidates are given as a compact list: ID | TYPE | DESCRIPTION.")
        else:
            lines.append("Candidates are given as JSON objects with id, name, description, and types.")
        lines.append("")

        for task in tasks:
            formatted_table = self._format_table(task["table"])
            entity_mention = task.get("entity_mention") or "N.A."
            candidates = task.get("candidates") or []
            candidates_text = self._format_candidates_text(
                candidates,
                compact=compact,
                format_candidates=format_candidates,
            )

            lines.append("### Task")
            lines.append(f'TASK_ID: "{task["task_id"]}"')
            lines.append("TABLE_ROW:")
            lines.append(formatted_table)
            lines.append("ENTITY_MENTION:")
            lines.append(str(entity_mention))
            lines.append("CANDIDATES:")
            lines.append(candidates_text)
            lines.append("")

        lines.append(answer_format)
        return "\n".join(lines)
