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
        if table_metadata is None:
            table_metadata = "N.A."
        else:
            table_metadata = " ".join(table_metadata.split())
        if table_summary is None:
            table_summary = "N.A."
        else:
            table_summary = " ".join(table_summary.split())
        if column_name is None:
            column_name = "N.A."

        # Optimize candidates list by reducing the verbosity of the JSON representation
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

        if format_candidates:
            if not self.tablellama_format:
                candidates_text = ",".join(
                    [
                        f"<id: {candidate['id']}; "
                        f"name: {candidate['name']}; "
                        f"description: {candidate['description'] if candidate['description'] is not None else 'N.A.'}; "  # noqa: E501
                        f"types: {','.join([t['name'] for t in candidate['types'] if t['name'] is not None])}>"  # noqa: E501
                        for candidate in optimized_candidates
                    ]
                )
            else:
                candidates_text = ",".join(
                    [
                        f"<{candidate['name']} "
                        f"[DESCRIPTION] {candidate['description'] if candidate['description'] is not None else 'None'} "  # noqa: E501
                        f"[TYPE] {','.join([t['name'] for t in candidate['types'] if t['name'] is not None])}>"  # noqa: E501
                        for candidate in optimized_candidates
                    ]
                )
        else:
            if compact:
                # Convert optimized candidates list to a compact JSON string
                candidates_text = json.dumps(optimized_candidates, separators=(",", ":"))
            else:
                # Convert optimized candidates list to a pretty-printed JSON string
                candidates_text = json.dumps(optimized_candidates, indent=2)

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
