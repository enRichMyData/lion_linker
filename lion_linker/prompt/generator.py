from __future__ import annotations

import copy
import json


class PromptGenerator:
    def __init__(self, prompt_file):
        with open(prompt_file, "r") as file:
            self.template = file.read()

    def _format_table(self, table: list[list[str]]) -> str:
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
                "types": [{"id": t["id"], "name": t["name"]} for t in candidate["types"]],
            }
            optimized_candidates.append(optimized_candidate)

        if format_candidates:
            candidates_text = ", ".join(
                [
                    f"<[id] {candidate['id']}; "
                    f"[name] {candidate['name']}; "
                    f"[description] {candidate['description']}; "
                    f"[types] {','.join([t['name'] for t in candidate['types']])}>"
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
            "[TABLE]": formatted_table,
            "[TABLE METADATA]": table_metadata,
            "[SUMMARY]": table_summary,
            "[COLUMN NAME]": column_name,
            "[ENTITY MENTION]": entity_mention,
            "[CANDIDATES]": candidates_text,
        }

        # Replace each placeholder using the dictionary
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, str(value))

        return template
