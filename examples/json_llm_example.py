from pydantic import BaseModel, Field
import util
import argparse

# Parse the sentence from the command line
def args():
    parser = argparse.ArgumentParser(description='Test Fruit Extractor JSON LLM Call')
    parser.add_argument('sentence', type=str, help='The sentence to send to the LLM')
    return parser.parse_args()

# Define the JSON schema for the LLM
# { "ContainsFruit": false, "ContainsVegetable": false, "ContainsMeat": false, "ContainsDairy": false }
class FoodExtractor(BaseModel):
    ContainsFruit: bool = Field(default=False)
    ContainsVegetable: bool = Field(default=False)
    ContainsMeat: bool = Field(default=False)
    ContainsDairy: bool = Field(default=False)

# Call the LLM with the sentence and the JSON schema
def extract_food(sentence: str):
    system_prompt = "You are a food extractor bot for organizing recipes.  Read the sentence and extract the food into a JSON object."
    message = sentence
    json_class = FoodExtractor

    response = util.json_llm_call(system_prompt, message, json_class)

    return response

if __name__ == '__main__':
    extracted_food = extract_food(args().sentence)

    print()
    print("Sentence: " + args().sentence)   
    try:   
        print("ContainsFruit: " + str(extracted_food['ContainsFruit']))
        print("ContainsVegetable: " + str(extracted_food['ContainsVegetable']))
        print("ContainsMeat: " + str(extracted_food['ContainsMeat']))
        print("ContainsDairy: " + str(extracted_food['ContainsDairy']))
    except:
        print("WARNING: The LLM response does not align with the input schema. You may call the API again or handle the issue accordingly.")

    print()
    print(extracted_food)

# Example usage:
# python json_llm_example.py "I like to eat apples and bananas"
# python json_llm_example.py "I like to eat steak and potatoes"