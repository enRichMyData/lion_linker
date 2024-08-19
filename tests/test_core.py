import pytest
from lion_linker.core import PromptGenerator

def test_generate_prompt():
    prompt_file = 'test_prompt_template.txt'
    with open(prompt_file, 'w') as file:
        file.write(
            "Here is the summary of the table:\n[TABLE SUMMARY]\n\n"
            "Here is the data for the current row:\n[ROW]\n\n"
            "The column name in question is:\n[COLUMN NAME]\n\n"
            "The entity mention is:\n[ENTITY MENTION]\n\n"
            "Possible candidates for the entity are:\n[CANDIDATES]\n\n"
            "Please provide only the QID. If no correct candidate is present, provide NIL."
        )
    
    prompt_generator = PromptGenerator(prompt_file)
    table_summary = "Summary of the movies table."
    row_data = {'title': 'The Matrix', 'year': 1999, 'director': 'The Wachowskis'}
    column_name = 'title'
    entity_mention = 'The Matrix'
    candidates = 'The Matrix, The Matrix Reloaded'

    prompt_text = prompt_generator.generate_prompt(table_summary, row_data, column_name, entity_mention, candidates)
    
    assert "Here is the summary of the table:" in prompt_text
    assert "Summary of the movies table." in prompt_text
    assert "Here is the data for the current row:" in prompt_text
    assert "{'title': 'The Matrix', 'year': 1999, 'director': 'The Wachowskis'}" in prompt_text
    assert "The column name in question is:" in prompt_text
    assert "title" in prompt_text
    assert "The entity mention is:" in prompt_text
    assert "The Matrix" in prompt_text
    assert "Possible candidates for the entity are:" in prompt_text
    assert "The Matrix, The Matrix Reloaded" in prompt_text