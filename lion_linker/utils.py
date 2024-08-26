
def process_in_batches(df, batch_size, process_function):
    num_batches = (len(df) + batch_size - 1) // batch_size
    results = []
    for i in range(num_batches):
        batch_df = df[i * batch_size:(i + 1) * batch_size]
        results.extend(process_function(batch_df))
    return results

def parse_response(response_json):
    result = []
    for item in response_json:
        entity_id = item.get('id')
        entity_name = item.get('name')
        description = item.get('description')
        types = item.get('types')
        result.append({
            'id': entity_id,
            'name': entity_name,
            'description': description,
            'types': types
        })
    return result