```json
{
  "datasetId": "c493ef7fdeb844048bb444e0a012c3fb",
  "tableId": "26637898e90c484bcd211db506ea0fe34",
  "jobId": "e253a2fb777c4cc8b016c321a922cfb5",
  "status": "completed",
  "page": 1,
  "perPage": 50,
  "totalRows": 2,
  "rows": [
    {
      "idRow": 0,
      "data": [
        "OpenAI",
        "United States"
      ],
      "predictions": [
        {
          "column": "buyer",
          "answer": {
            "candidate_ranking": [
              {
                "id": "Q124605186",
                "confidence_label": "HIGH",
                "confidence_score": 0.92,
                "name": "OpenAI",
                "types": [{"id": "", "name": "company"}],
                "description": "US artificial intelligence lab",
                "match": true
              },
              {
                "id": "Q21708200",
                "confidence_label": "MEDIUM",
                "confidence_score": 0.63,
                "name": "OpenAI Inc.",
                "types": [{"id": "", "name": "nonprofit organization"}],
                "description": "Non-profit parent company of OpenAI",
                "match": false
              },
              {
                "id": "Q116374474",
                "confidence_label": "LOW",
                "confidence_score": 0.31,
                "name": "OpenAI LP",
                "types": [{"id": "", "name": "technology company"}],
                "description": "OpenAI commercial subsidiary",
                "match": false
              },
              {
                "id": "Q124605344",
                "confidence_label": "LOW",
                "confidence_score": 0.24,
                "name": "OpenAI Charter",
                "types": [{"id": "", "name": "document"}],
                "description": "OpenAI policy document",
                "match": false
              },
              {
                "id": "Q124734193",
                "confidence_label": "LOW",
                "confidence_score": 0.18,
                "name": "OpenAI Safety",
                "types": [{"id": "", "name": "organization"}],
                "description": "Safety research group",
                "match": false
              }
            ],
            "explanation": "‘OpenAI’ in the table clearly maps to the US AI lab; other candidates describe related but distinct entities."
          },
          "identifier": "Q124605186"
        }
      ]
    }
  ],
  "annotationMeta": {
    "jobId": "e253a2fb777c4cc8b016c321a922cfb5",
    "updatedAt": "2025-05-01T10:32:09.000000",
    "lionConfig": {
      "model_name": "gpt-oss:20b",
      "chunk_size": 64,
      "table_ctx_size": 1
    },
    "retrieverConfig": {
      "class_path": "lion_linker.retrievers.LamapiClient",
      "kg": "wikidata"
    }
  }
}
```
