import util
import argparse

# Parse the sentence from the command line
def args():
    parser = argparse.ArgumentParser(description='Test Sentiment Flipper LLM Call')
    parser.add_argument('sentence', type=str, help='The sentence to send to the LLM')
    return parser.parse_args()

# Call the LLM with the sentence
def flip_sentiment(sentence: str):
    system_prompt = """You are a sentiment flipper bot for helping us collect inverted sentiment data.""" +\
    """Read the user utterance and respond with a statement of exactly the opposite sentiment, """ +\
    """but otherwise precisely the same content.  Respond with just the inverted statement and nothing more.\n\n"""

    message = sentence

    # Our llm will stop when it sees a newline character.
    # You can add more stop tokens to the list if you want to stop on other tokens!
    # Feel free to remove the stop parameter if you want the llm to run to completion.
    stop = ["\n"]

    response = util.simple_llm_call(system_prompt, message, stop=stop)

    return response

if __name__ == '__main__':
    flipped_sentiment = flip_sentiment(args().sentence)

    print()
    print("Sentence: " + args().sentence)   

    print("Flipped Sentiment: " + flipped_sentiment)

# Example usage:
# python simple_llm_example.py "I am happy today"
# python simple_llm_example.py "What a horrible noise!"