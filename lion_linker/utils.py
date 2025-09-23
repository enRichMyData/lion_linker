from importlib import resources


def parse_response(response_json):
    result = []
    for item in response_json:
        entity_id = item.get("id")
        entity_name = item.get("name")
        description = item.get("description")
        types = item.get("types")
        result.append(
            {"id": entity_id, "name": entity_name, "description": description, "types": types}
        )
    return result


def load_prompt(name: str) -> str:
    with (
        resources.files("lion_linker.prompt")
        .joinpath("prompt_template_" + name + ".txt")
        .open("r", encoding="utf-8") as f
    ):
        return f.read()
