#!/usr/bin/env python

# PA7, CS124, Stanford
# v.1.0.4
#
# Usage:
#   python sanity_check.py --help
######################################################################
import inspect
import os
import sys
import argparse
import numpy as np
import math

# Add the parent directory to PATH so we can import chatbot
# from the parent directory
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from chatbot import Chatbot


def assert_numpy_array_equals(givenValue, correctValue, failureMessage):
    try:
        assert np.array_equal(givenValue, correctValue)
        return True
    except AssertionError:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False


def assert_list_equals(givenValue, correctValue, failureMessage,
                       orderMatters=True):
    try:
        if orderMatters:
            assert givenValue == correctValue
            return True
        givenValueSet = set(givenValue)
        correctValueSet = set(correctValue)
        assert givenValueSet == correctValueSet
        return True
    except AssertionError:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False
    
def assert_list_contains_and_excludes(givenSet, correctSet, incorrectSet, failureMessage):
    try:
        assert set(givenSet).issuperset(correctSet)
        assert set(givenSet).isdisjoint(incorrectSet)
        return True
    except AssertionError:
        print(failureMessage)
        print("Expected: {}".format(correctSet))
        print("Actual: {}".format(givenSet))
        print("Must not contain: {}".format(incorrectSet))
        return False

def assert_sign_equals(givenValue, correctValue, failureMessage):
    try:
        if abs(givenValue) > 0:
            givenValue = int(givenValue / abs(givenValue))
        assert givenValue == correctValue
        return True
    except AssertionError:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False


def test_similarity():
    print("Testing similarity() functionality...")
    chatbot = Chatbot(False)

    x = np.array([1, 1, -1, 0], dtype=float)
    y = np.array([1, 0, 1, -1], dtype=float)

    self_similarity = chatbot.similarity(x, x)
    if not math.isclose(self_similarity, 1.0):
        print('Unexpected cosine similarity between {} and itself'.format(x))
        print('Expected 1.0, calculated {}'.format(self_similarity))
        print()
        return False

    ortho_similarity = chatbot.similarity(x, y)
    if not math.isclose(ortho_similarity, 0.0):
        print('Unexpected cosine similarity between {} and {}'.format(x, y))
        print('Expected 0.0, calculated {}'.format(ortho_similarity))
        print()
        return False

    print('similarity() sanity check passed!')
    print()
    return True


def test_binarize():
    print("Testing binarize() functionality...")
    chatbot = Chatbot(False)
    if assert_numpy_array_equals(
            chatbot.binarize(np.array([[1, 2.5, 5, 0]])),
            np.array([[-1., -1., 1., 0.]]),
            "Incorrect output for binarize(np.array([[1, 2.5, 5, 0]]))."
    ):
        print("binarize() sanity check passed!")
    print()


def test_extract_titles():
    print("Testing extract_titles() functionality...")
    chatbot = Chatbot(False)

    # add more test cases here!!!
    test_cases = [
        ('I liked "The Notebook"', ["The Notebook"]),
        ('You are a great bot!', []),
        ('I enjoyed "Titanic (1997)" and "Scream 2 (1997)"',
         ["Titanic (1997)", "Scream 2 (1997)"]),
    ]

    tests_passed = True
    for input_text, expected_output in test_cases:
        if not assert_list_equals(
                chatbot.extract_titles(chatbot.preprocess(input_text)),
                expected_output,
                "Incorrect output for extract_titles("
                "chatbot.preprocess('{}')).".format(
                    input_text),
                orderMatters=False
        ):
            tests_passed = False
    if tests_passed:
        print('extract_titles() sanity check passed!')
    print()


def test_find_movies_by_title():
    print("Testing find_movies_by_title() functionality...")
    chatbot = Chatbot(False)

    # add more test cases here!!!
    test_cases = [
        ('The American President', [10]),
        ('Titanic', [1359, 2716]),
        ('Titanic (1997)', [1359]),
        ('An American in Paris (1951)', [721]),
        ('The Notebook (1220)', []),
        ('Scream', [1142]),
    ]

    tests_passed = True
    for input_text, expected_output in test_cases:
        if not assert_list_equals(
                chatbot.find_movies_by_title(input_text),
                expected_output,
                "Incorrect output for find_movies_by_title('{}').".format(
                    input_text),
                orderMatters=False
        ):
            tests_passed = False
    if tests_passed:
        print('find_movies_by_title() sanity check passed!')
    print()

def test_find_movies_by_foreign_title():
    print("Testing find_movies_by_title() foreign film functionality... This might take a moment if you use LLM JSON outputs!")
    chatbot = Chatbot(True)

    # These test cases are foreign titles not listed in the movies.txt file
    # They are translations of the English titles found in the movies.txt file
    test_cases_translations = [
        ('Jernmand', [6944]), # Danish
        ('Un Roi à New York', [2906]), # French
        ('Tote Männer Tragen Kein Plaid', [1670]), # German
        ('Indiana Jones e il Tempio Maledetto', [1675]), # Italian
        ('Junglebogen', [326, 1638, 8947]), # Danish
        ('Doble Felicidad', [306]), # Spanish
        ('Der König der Löwen', [328]), # German
    ]

    tests_passed = True
    for input_text, expected_output in test_cases_translations:
        if not assert_list_equals(
                chatbot.find_movies_by_title(input_text),
                expected_output,
                "Incorrect output for find_movies_by_title('{}').  Note this movie was a translation of an English title in the movies.txt file.".format(
                    input_text),
                orderMatters=False
        ):
            tests_passed = False

    if tests_passed:
        print('find_movies_by_title() foreign film title sanity check passed!')
    print()


def test_extract_sentiment():
    print("Testing extract_sentiment() functionality...")
    chatbot = Chatbot(False)

    # add more test cases here!!!
    test_cases = [
        ('I like "Titanic (1997)".', 1),
        ('I saw "Titanic (1997)".', 0),
        ('I didn\'t enjoy "Titanic (1997)".', -1),
        ('I didn\'t really like "Titanic (1997)".', -1),
        ('I never liked "Titanic (1997)".', -1),
        ('I really enjoyed "Titanic (1997)".', 1),
        ('"Titanic (1997)" started out terrible, but the ending was totally '
         'great and I loved it!', 1),
        ('I loved "10 Things I Hate About You"', 1),
    ]

    tests_passed = True
    for input_text, expected_output in test_cases:
        if not assert_sign_equals(
                chatbot.extract_sentiment(chatbot.preprocess(input_text)),
                expected_output,
                "Incorrect output for extract_sentiment(chatbot.preprocess('"
                "{}')).".format(
                    input_text)
        ):
            tests_passed = False
    if tests_passed:
        print('extract_sentiment() sanity check passed!')
    print()

def test_recommend():
    print("Testing recommend() functionality...")
    chatbot = Chatbot(False)

    user_ratings = np.array([1, -1, 0, 0, 0, 0])
    all_ratings = np.array([
        [1, 1, 1, 0],
        [1, -1, 0, -1],
        [1, 1, 1, 0],
        [0, 1, 1, -1],
        [0, -1, 1, -1],
        [-1, -1, -1, 0],
    ])
    small_recommendations = chatbot.recommend(user_ratings, all_ratings, 2)
    user_ratings = np.zeros(9125)
    user_ratings[[8514, 7953, 6979, 7890]] = 1
    user_ratings[[7369, 8726]] = -1
    recommendations = chatbot.recommend(user_ratings, chatbot.ratings, k=5)

    test_cases = [
        (small_recommendations, [2, 3]),
        (recommendations, [8582, 8596, 8786, 8309, 8637]),
    ]

    tests_passed = True
    for i, (recs, expected_output) in enumerate(test_cases):
        if not assert_list_equals(
                recs,
                expected_output,
                "Test case #{} for recommender tests failed".format(i),
        ):
            tests_passed = False
    if tests_passed:
        print('recommend() sanity check passed!')
    print()

def test_extract_emotion():
    print("Testing extract_emotion() functionality... (This might take a moment if you use LLM JSON Outputs!)")
    chatbot = Chatbot(True)

    # add more test cases here!!!
    test_cases = [
        ('I am quite frustrated by these awful recommendations!!!', set(["anger"])),
        ('Great suggestion!  It put me in a great mood!', set(['happiness'])),
        ('Disgusting!!!', set(["disgust"])),
        ('Woah!!  That movie was so shockingly bad!  You had better stop making awful recommendations they\'re pissing me off.', set(["anger", "surprise"])),
        ('What movie would you suggest I watch next?', set([])),
        ('What\'s the weather?', set([])),
        ('Ack, woah!  Oh my gosh, what was that?  Really startled me.  I just heard something really frightening!', set(["fear", "surprise"])),
        ('I am so sad.  I just watched the saddest movie ever.', set(["sadness"])),
        ('Well, that was a delightful movie!', set(["happiness"])),
    ]
    all_emotions = {"anger", "disgust", "fear", "happiness", "sadness", "surprise"}

    tests_passed = True
    for input_text, expected_output in test_cases:
        if not assert_list_contains_and_excludes(
                set([emotion.lower() for emotion in chatbot.extract_emotion(chatbot.preprocess(input_text))]),
                expected_output,
                all_emotions - expected_output,
                "Incorrect output for extract_emotion(chatbot.preprocess('"
                "{}')).".format(
                    input_text)
        ):
            tests_passed = False
    
    if tests_passed:
        print('extract_emotion() sanity check passed!')
    print()

def main():
    parser = argparse.ArgumentParser(
        description='Sanity checks the chatbot. If no arguments are passed, all'
                    ' checks for starter mode are run; you can use the '
                    'arguments below to test specific parts of the '
                    'functionality.')

    parser.add_argument('-a', '--all', help='Tests all of the functions',
                        action='store_true')
    parser.add_argument('-c', '--llm_programming',
                        help='Tests all of the llm programming functions',
                        action='store_true')
    parser.add_argument('--extract-titles',
                        help='Tests only the extract_titles function',
                        action='store_true')
    parser.add_argument('--find-movies',
                        help='Tests only the find_movies_by_title function',
                        action='store_true')
    parser.add_argument('--extract-sentiment',
                        help='Tests only the extract_sentiment function',
                        action='store_true')
    parser.add_argument('--recommend', help='Tests only the recommend function',
                        action='store_true')
    parser.add_argument('--binarize', help='Tests only the binarize function',
                        action='store_true')
    parser.add_argument('--similarity',
                        help='Tests only the similarity function',
                        action='store_true')
    parser.add_argument('--extract_emotion',
                        help='Tests only the extract_emotion function',
                        action='store_true')
    parser.add_argument('--foreign_title',
                        help='Tests only the find_movies_by_title function with foreign titles',
                        action='store_true')

    args = parser.parse_args()
    if args.extract_titles:
        test_extract_titles()
        return
    if args.find_movies:
        test_find_movies_by_title()
        return
    if args.extract_sentiment:
        test_extract_sentiment()
        return
    if args.recommend:
        test_recommend()
        return
    if args.binarize:
        test_binarize()
        return
    if args.similarity:
        test_similarity()
        return
    if args.extract_emotion:
        test_extract_emotion()
        return
    if args.foreign_title:
        test_find_movies_by_foreign_title()
        return

    testing_llm_programming = args.llm_programming
    testing_all = args.all

    if not testing_llm_programming or testing_all:
        test_extract_titles()
        test_find_movies_by_title()
        test_extract_sentiment()
        # comment out test_recommend() if it's taking too long!
        test_recommend()
        test_binarize()
        test_similarity()

    if testing_llm_programming or testing_all:
        test_extract_emotion()
        test_find_movies_by_foreign_title()


if __name__ == '__main__':
    main()
