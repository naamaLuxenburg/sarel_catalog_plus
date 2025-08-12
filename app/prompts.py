#constants for prompts

input_shortcut='input_shortcut'
gpt_meaning='gpt_meaning'
gpt_desc_match='gpt_desc_match'
gpt_accuracy='gpt_accuracy'
gpt_meaning_alternatives='gpt_meaning_alternatives'
reasoning='reasoning'

gpt_alter_meaning='meaning'
gpt_alter_desc_match='desc_match'
gpt_alter_accuracy='accuracy'


model_name_abbreviations="gpt-4"

system_message_abbreviations = """
    You are a professional medical equipment analyst.
    You specialize in interpreting abbreviations found in medical equipment and pharmaceutical descriptions.
    Always return your analysis as valid JSON. Include your best guess, supporting indices, and confidence level.
    """

user_message_abbreviations = """
You are given a list of shortcuts with example descriptions. For each shortcut:
1. Interpret its most likely full word.
2. Indicate which example descriptions support your interpretation (by their number).
3. Give a confidence score from 0 to 1.
4. If other meanings are also valid for different descriptions, list each of them with:
* The alternative meaning (e.g., 'dispense', 'display')
* The matching description indices
* A separate confidence score
5. Ensure that all the example descriptions are covered by either the main meaning or an alternative.
6. add a short reasoning field explaining your interpretation and logic.

Use this JSON format for each:
{
  "input_shortcut": "disp",
  "gpt_meaning": "disposable",
  "gpt_desc_match": [0, 1, 2],
  "gpt_accuracy": 0.97,
  "gpt_meaning_alternatives": [
    {
      "meaning": "display",
      "desc_match": [3],
      "accuracy": 0.65
    },
    {
      "meaning": "dispense",
      "desc_match": [4, 5],
      "accuracy": 0.6
    }
  ],
  "reasoning": "Descriptions 0–2 refer to gloves and syringes, which are typically 'disposable'. Index 3 mentions screen-related terms, suggesting 'display'. Index 4 and 5 mention drug delivery systems, suggesting 'dispense' as another possible meaning."
}


Only include *_other fields if truly relevant.

Examples:
"""




