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


def test_extract_sentiment_for_movies():
    print("Testing test_extract_sentiment_for_movies() functionality...")
    chatbot = Chatbot(True)

    # add more test cases here!!!
    test_cases = [
        ('I liked both "I, Robot" and "Ex Machina".',
         [("I, Robot", 1), ("Ex Machina", 1)]),
        ('I liked "I, Robot" but not "Ex Machina".',
         [("I, Robot", 1), ("Ex Machina", -1)]),
        ('I didn\'t like either "I, Robot" or "Ex Machina".',
         [("I, Robot", -1), ("Ex Machina", -1)]),
        ('I liked "Titanic (1997)", but "Ex Machina" was not good.',
         [("Titanic (1997)", 1), ("Ex Machina", -1)]),
    ]

    tests_passed = True
    for input_text, expected_output in test_cases:
        if not assert_list_equals(
                chatbot.extract_sentiment_for_movies(
                    chatbot.preprocess(input_text)),
                expected_output,
                "Incorrect output for extract_sentiment_for_movies("
                "chatbot.preprocess('{}')).".format(
                    input_text),
                orderMatters=False
        ):
            tests_passed = False
    if tests_passed:
        print('extract_sentiment_for_movies() sanity check passed!')
    print()


def test_find_movies_closest_to_title():
    print("Testing find_movies_closest_to_title() functionality...")
    chatbot = Chatbot(True)

    # add more test cases here!!!
    test_cases = [
        ('Sleeping Beaty', [1656]),
        ('Te', [8082, 4511, 1664]),
        ('BAT-MAAAN', [524, 5743]),
        ('Blargdeblargh', []),
    ]

    tests_passed = True
    for input_text, expected_output in test_cases:
        if not assert_list_equals(
                chatbot.find_movies_closest_to_title(input_text),
                expected_output,
                "Incorrect output for find_movies_closest_to_title("
                "chatbot.preprocess('{}')).".format(
                    input_text),
                orderMatters=False
        ):
            tests_passed = False
    if tests_passed:
        print('find_movies_closest_to_title() sanity check passed!')
    print()
    return True


def test_disambiguate():
    print("Testing disambiguate() functionality...")
    chatbot = Chatbot(True)

    # add more test cases here!!!
    test_cases = [
        ('1997', [1359, 2716], [1359]),
        ('2', [1142, 1357, 2629, 546], [1357]),
        ('Sorcerer\'s Stone', [3812, 4325, 5399, 6294, 6735, 7274, 7670, 7842],
         [3812]),
    ]

    tests_passed = True
    for clarification, candidates, expected_output in test_cases:
        if not assert_list_equals(
                chatbot.disambiguate(clarification, candidates),
                expected_output,
                "Incorrect output for disambiguate('{}', {})".format(
                    clarification, candidates),
                orderMatters=False
        ):
            tests_passed = False
    if tests_passed:
        print('disambiguate() sanity check passed!')
    print()
    return True


def test_disambiguate_complex():
    print("Testing complex disambiguate() functionality...")
    chatbot = Chatbot(True)

    # add more test cases here!!!
    test_cases = [
        ('2', [8082, 4511, 1664], [4511]),
        ('most recent', [524, 5743], [524]),
        ('the Goblet of Fire one',
         [3812, 4325, 5399, 6294, 6735, 7274, 7670, 7842], [6294]),
        ('the second one', [3812, 6294, 4325, 5399, 6735, 7274, 7670, 7842],
         [6294]),
    ]

    tests_passed = True
    for clarification, candidates, expected_output in test_cases:
        if not assert_list_equals(
                chatbot.disambiguate(clarification, candidates),
                expected_output,
                "Incorrect output for complex disambiguate('{}', {})".format(
                    clarification, candidates),
                orderMatters=False
        ):
            tests_passed = False
    if tests_passed:
        print('complex disambiguate() sanity check passed!')
    print()
    return True


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


def main():
    parser = argparse.ArgumentParser(
        description='Sanity checks the chatbot. If no arguments are passed, all'
                    ' checks for starter mode are run; you can use the '
                    'arguments below to test specific parts of the '
                    'functionality.')

    parser.add_argument('-a', '--all', help='Tests all of the functions',
                        action='store_true')
    parser.add_argument('-c', '--creative',
                        help='Tests all of the creative function',
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
    parser.add_argument('--find-closest',
                        help='Tests only the find_movies_closest_to_title '
                             'function',
                        action='store_true')
    parser.add_argument('--extract-sentiment-multiple',
                        help='Tests only the extract_sentiment_for_movies '
                             'function',
                        action='store_true')
    parser.add_argument('--disambiguate',
                        help='Tests only the disambiguate functions (for part '
                             '2 and 3)',
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
    if args.find_closest:
        test_find_movies_closest_to_title()
        return
    if args.extract_sentiment_multiple:
        test_extract_sentiment_for_movies()
        return
    if args.disambiguate:
        test_disambiguate()
        test_disambiguate_complex()
        return

    testing_creative = args.creative
    testing_all = args.all

    if not testing_creative or testing_all:
        test_extract_titles()
        test_find_movies_by_title()
        test_extract_sentiment()
        # comment out test_recommend() if it's taking too long!
        test_recommend()
        test_binarize()
        test_similarity()

    if testing_creative or testing_all:
        test_find_movies_closest_to_title()
        test_extract_sentiment_for_movies()
        test_disambiguate()
        test_disambiguate_complex()


if __name__ == '__main__':
    main()
