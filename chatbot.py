# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util
from collections import defaultdict
import numpy as np
# from nltk.tokenize import word_tokenize # need to delete
import regex as re 
from porter_stemmer import PorterStemmer
from itertools import combinations


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'ELON'

        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)

        self.user_ratings = np.zeros((len(ratings)))

        self.input_count = 0
        self.reachedRecommendation = False
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Hello, I'm ELON! I'm going to help you find a movie to watch. Tell me about a movie you watched and whether you liked the movie. Please put the name of the movie in quotation marks."

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "I hope you have a great day and remember to full send!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################
    def tokenize(self, raw_text):
        """
        Tokenizes raw_text by seperating the input string by puncuation. Converts contractions
        into full words. Returns a list of tokens
        """
        tokens = re.findall(r"\w+", raw_text)
        for x in range(len(tokens)):
            if tokens[x] == 's':
                tokens[x] = 'is'
            elif tokens[x] == 't':
                tokens[x] = 'not'
            elif tokens[x] == 're':
                tokens[x] = 'are'
        return tokens

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.creative:
            response = "I processed {} in creative mode!!".format(line)
        else:
            # In starter mode, your chatbot will help the user by giving movie recommendations. 
            # It will ask the user to say something about movies they have liked, and it will come up with a recommendation based on those data points. 
            # The user of the chatbot in starter mode will follow its instructions, so you will deal with fairly restricted input. 
            # Movie names will come in quotation marks and expressions of sentiment will be simple.
            movieTitle = self.extract_titles(self.preprocess(line))
            sentiment = self.extract_sentiment(line)

            if len(movieTitle) == 0: # added this check for no movie, 
                #what about the case where mentioned movie is not in the database??
                response = "Please put quotation marks around the movie name so that I can tell what movie you are talking about."
                return response
            elif len(movieTitle) >= 2: 
                response = "Please tell me about one movie at a time. What's one movie you have seen?"
                return response
            else:
                movieIdx = self.find_movies_by_title(movieTitle[0]) # added [0] here
                if len(movieIdx) == 0:
                    return "Huh, I\'m not sure what movie you are talking about. What's a different movie that you have seen?"
                elif len(movieIdx) >= 2:
                    return "I found more than one movie called " + movieTitle[0] + ". Which one were you trying to tell me about?"
                else:
                    if self.user_ratings[movieIdx] == 0 and sentiment != 0:
                        self.input_count += 1
                    self.user_ratings[movieIdx] = sentiment

                    if self.input_count == 5:
                        self.user_ratings = self.binarize(self.user_ratings)
                        self.ratings = self.binarize(self.ratings)
                        # print(len(self.titles)) #9125
                        # print(self.user_ratings.shape) #(9125, )
                        # print(self.ratings.shape) #(9125, 671)
                        
                         #get number of movies that the user haven't watched, and pass it in as k to recommend?
                        k = len([rating for rating in self.user_ratings if rating == 0])
                        recommendations = self.recommend(self.user_ratings, self.ratings, k, creative=False) #prior k: len(self.titles) - 1.  at most, k will be number of movies
                        i = -1
                        affirmative = ["yes", "sure", "ok", "yeah", "y", "affirmative", "i guess so", "fine", "always"]
                        negative = ["no", "nah", "never", "negative", "n", "no thanks", "no, thanks", "nope"]
                        answer = "yes" 
                        while (answer in affirmative):
                            i += 1
                            answer = input('I think you\'ll enjoy watching\"' + self.titles[recommendations[i]][0] + '\"! Would you like another recommendations?\n').lower()
                            if answer in negative:
                                # response = "Have a nice day. Fullsend!"
                                break
                            elif answer not in affirmative and answer not in negative:
                                currInput = input("Please input \"Yes\" or \"No\". ELON is disappointed in you. Let's try again. Would you like more recommendations?\n").lower()
                                while (currInput != 'yes' and currInput != 'no'):
                                    currInput = input("Please input \"Yes\" or \"No\". ELON is disappointed in you. Let's try again. Would you like more recommendations?\n")
                                answer = currInput
                                
                            if i == len(self.titles):
                                response = "We have no more recommendations -- Have a nice day. Fullsend!"
                    else: #if self.input_count < 5
                        if sentiment == 1:
                            return "Ok, you liked \"" + movieTitle[0] + "\"! Tell me what you thought of another movie."
                        elif sentiment == -1:
                            return "Ok, you didn't like \"" + movieTitle[0] + "\"! Tell me what you thought of another movie."
                        else: 
                            self.input_count -= 1
                            return "I'm confused, did you like \"" + movieTitle[0] + "\"? Please try to clarify if you liked the movie."
            response = self.goodbye()
            # response = "I processed {} in starter mode!!".format(line)

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def phrase_builder(self, my_words): # https://stackoverflow.com/questions/24964862/printing-all-possible-phrases-consecutive-combinations-of-words-in-a-given-str/24965141
        phrases = []
        for i, word in enumerate(my_words):
            phrases.append(word)
            for nextword in my_words[i+1:]:
                phrases.append(phrases[-1] + " " + nextword)
                # Remove the one-word phrase.
                if word in phrases:
                    phrases.remove(word)
        return phrases

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.
        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """ 
        if self.creative:
            return re.findall('"([^"]*)"', preprocessed_input)
        else:
            movie_titles = []
            result = []
            for title in self.titles:
                movie_titles.append(title[0].lower())
                # if "notebook" in title[0].lower():
                #     print(title[0].lower())
                movie_titles.append(re.sub("( \([0-9]+\))", "", title[0]).lower()) # add both the title w/ year and without to list
            allMovies = []
            
            preprocessed_input = re.sub("(\")", "", preprocessed_input)
            words = preprocessed_input.split()
            substrings = self.phrase_builder(words)
            # print(substrings)
            for phrase in substrings:
                if phrase.lower() in movie_titles:
                    result.append(phrase)
            result_sorted = sorted(result)
            # print(result_sorted)
            for i in range(1, len(result_sorted)):
                if i < len(result_sorted):
                    i1 = result_sorted[i]
                    i2 = result_sorted[i - 1]
                    if re.sub( "(\([0-9]+\))", "", i1).strip() == re.sub( "(\([0-9]+\))", "", i2).strip():
                        result.remove(result_sorted[i - 1]) # remove the one without the year, it passes scream/titanic
                        result_sorted.pop(i)
                        
            return result

        # for movie in substring:
        #     if movie in self.titles

        # allMovies = []
        # for movie in self.titles:
        #     movie = movie[0]
        #     movieTitle = re.sub("(\([0-9]+\))", "", movie)
        #     if movieTitle in preprocessed_input and movieTitle not in allMovies:
        #         allMovies.append(movie)
        # return allMovies

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        res = []
        titleYear = re.findall("(\([0-9]+\))", title) 
        title = re.sub( "(\([0-9]+\))", "", title) # filter out year from original title
        titleWords = self.tokenize(title)
        # tokenize movie title, mamke sure every token in movie title input also in tokenized representation of movie
        for i in range(len(self.titles)):
            currTitle = self.titles[i][0]
            if currTitle == title:
                res.append(i)
            else:
                currYear = re.findall("(\([0-9]+\))", currTitle)
                currTitle = re.sub("(\([0-9]+\))", "", currTitle)
                currWords = self.tokenize(currTitle)
                sameMovie = True
                currWords = set(currWords)
                titleWords = set(titleWords)
                if ',' in currWords:
                    currWords.remove(',')
                if ',' in titleWords:
                    titleWords.remove(',')
                if currWords == titleWords: # the vectorized words are subsets of each other
                    if titleYear == [] or currYear[0] == titleYear[0]: # if there is a year specified in title make sure it matches
                        res.append(i) 
        return res

    # def levenshteinDistance(self, s1, s2): # https://stackoverflow.com/questions/2460177/edit-distance-in-python
    #     # print(s1)
    #     # print(s2)
    #     if len(s1) > len(s2):
    #         s1, s2 = s2, s1

    #     distances = range(len(s1) + 1)
    #     for i2, c2 in enumerate(s2):
    #         distances_ = [i2+1]
    #         for i1, c1 in enumerate(s1):
    #             if c1 == c2:
    #                 distances_.append(distances[i1])
    #             else:
    #                 distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
    #         distances = distances_
    #     return distances[-1]

    def extract_regex(self, s):
        if s == "loved":
            return "love"
        if s == "enjoyed":
            return "enjoy"
        s = re.sub( "(d)$", "", s) 
        s = re.sub( "(s)$", "", s) 
        s = re.sub( "(ing)$", "", s) 
        return s

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        pos_count = 0.0
        neg_count = 0.0
        lmd = 1.0
        neg_words = ["never", "not"]

        negated_flag = False
        distance_from_negation = 0
        max_negation_distance = 5
        preprocessed_input = re.sub('"([^"]*)"', "", preprocessed_input).lower()
        preprocessed_input = self.tokenize(preprocessed_input)
        for item in preprocessed_input:
            # print(item)
            if distance_from_negation == max_negation_distance:
                distance_from_negation = 0
                negated_flag = False
            if item not in self.sentiment:
                item = self.extract_regex(item)
            # print(item)
            if item != None and item in neg_words:
                negated_flag = not negated_flag
                continue
            if item not in self.sentiment:
                continue
            elif self.sentiment[item] != None and "pos" in self.sentiment[item]:
                if negated_flag:
                    neg_count += 1.0
                else:
                    pos_count += 1.0
            elif self.sentiment[item] != None and "neg" in self.sentiment[item]:
                if negated_flag:
                    pos_count += 1.0
                else:
                    neg_count += 1.0
            if negated_flag:
                distance_from_negation += 1
        if neg_count == 0.0 and pos_count == 0.0:
            return 0
        elif neg_count ==0:
            return 1
        if pos_count / neg_count > lmd:
            return 1
        elif pos_count / neg_count < lmd:
            return -1
        return 0

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        pass

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """

        pass

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        # new_candidates = []
        # for indice in candidates:
        #     if clarification in self.titles[indice][0] and indice not in new_candidates:
        #         new_candidates.append(indice)
        # titleYear = re.findall("(\([0-9]+\))", clarification) 
        # for indice in candidates:
        #     candidate_year = re.findall("(\([0-9]+\))", self.titles[indice][0])
        #     if indice not in new_candidates:
        #         if title_year != "" and titleYear == candidate_year:
        #             new_candidates.append(indice)
        #         elif clarification == candidate_year:
        #             new_candidates.append(indice)
        #         elif clarification == candidate_year[1:5]:
        # return new_candidates
        



    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        
        ratingsCopy = ratings.copy()
        binarized_ratings = np.where(ratings <= threshold, ratings, 1)
        binarized_ratings = np.where(ratings > threshold, binarized_ratings, -1)
        ratingsCopy = np.where(ratingsCopy != 0, binarized_ratings, 0)

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return ratingsCopy

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = np.dot(u,v)
        normalizing_factor = np.linalg.norm(u) * np.linalg.norm(v)
        if normalizing_factor == 0:
            similarity = 0
        else:
            similarity = np.divide(similarity, normalizing_factor)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.


        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []
        movieScore = {}
        rowLen = ratings_matrix.shape[0]
        # print(user_ratings)
        for i in range(rowLen):
            movieScore[i] = 0
            for j in range(rowLen):
                if i != j and user_ratings[j] != 0:
                    movieScore[i] += self.similarity(ratings_matrix[i], ratings_matrix[j]) * user_ratings[j]

        scoresToMovie = []
        for movie in movieScore:
            if user_ratings[movie] == 0: #if the user has not watched the movie 
                scoresToMovie.append((movieScore[movie], movie))
            
        scoresToMovie = sorted(scoresToMovie, reverse=True) #sorted(ratingsToMovie, key=lambda rating : rating[0], reverse=True)
        # print(len(scoresToMovie))
        # print(k)
        for idx in range(k):
            recommendations.append(scoresToMovie[idx][1])
        
        
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """

        # test for find_movies_by_title
        debug_info = 'debug info'
        id1 = "An American in Paris (1951)"
        # id2 = "The Notebook (1220)"
        # id3 = "Titanic"
        # id4 = "Scream"
        # l = list([id1])
        # for elem in l:
        #     print(elem, self.find_movies_by_title(elem))

        # test for extract_sentiment
        l2 = ["I didn't really like \"Titanic (1997)\"", "I never liked \"Titanic (1997)\"", "I really enjoyed \"Titanic (1997)\"",
                "I saw \"Titanic (1997)\".",  "\"Titanic (1997)\" started out terrible, but the ending was totally great and I loved it!",
                "I loved \"10 Things I Hate About You\""]
        for elem in l2:
            print(elem, self.extract_sentiment(elem))
        
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return "Hi! This is a temporary intro string."
        """
        Your task is to implement the chatbot as detailed in the PA6
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')

