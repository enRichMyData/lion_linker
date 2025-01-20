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
