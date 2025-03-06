# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
# Update: 2025-01 for Winter 2025 (Xuheng Cai)
######################################################################
from porter_stemmer import PorterStemmer
import util
from pydantic import BaseModel, Field

import numpy as np
import re
import random

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moviebot'

        self.llm_enabled = llm_enabled

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
        self.user_ratings = np.zeros(np.shape(self.ratings)[0])
        self.recommendations = []
        self.awaiting_recommendation = False
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

        greeting_message = "Hi! I'm a movie recommender bot. Tell me about a movie you've seen to get started!"

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

        goodbye_message = "Hope I was helpful. See ya!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

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
        if self.llm_enabled:
            response = "I processed {} in LLM Programming mode!!".format(line)
            return response
        else:
            affirmative_responses = [
                "yes", "yeah", "yep", "ya", "yup", "sure", "of course", "okay", "ok", 
                "alright", "absolutely", "certainly", "indeed", "affirmative", "right", 
                "for sure", "you bet", "totally", "sure thing", "aye", "roger that", "y", "yes please"
            ]
            negative_responses = ["no", "n", "nah", "nope", "not really", "no thanks", "no thank you"]
            
            unsure_responses = [
                "Hmm, I didn't quite catch that. Can you tell me about a movie you've seen recently?",
                "I'm not sure I recognized a movie in what you said. Could you mention a title?",
                "I couldn't identify a movie in your message. Let me know about one you've watched!"
            ]
            
            multiple_titles_responses = [
                "It looks like you mentioned multiple movies. Could you tell me about just one at a time?",
                "I see more than one movie in your response! Let's focus on one for now.",
                "You're talking about multiple movies! Pick one, and we can chat about it."
            ]
            
            unknown_movie_responses = [
                "Sorry, I couldn't find \"{}\" in my database. Could you try another movie?",
                "\"{}\" doesn't seem to be in my records. Maybe try another one?",
                "I couldn't locate \"{}\" in my database. Do you have another movie in mind?"
            ]
            
            multiple_matches_responses = [
                "I found multiple movies called \"{}\". Could you specify the year in parentheses along with the movie title?",
                "There are several movies named \"{}\". Please include the release year in parentheses in your response.",
                "\"{}\" has multiple versions! Try adding the year in parentheses to clarify which one you mean."
            ]
            
            neutral_sentiment_responses = [
                "I'm not sure how you feel about \"{}\". Could you clarify?",
                "Hmm, I can't tell if you liked \"{}\" or not. Could you let me know?",
                "It seems like you have neutral feelings about \"{}\". Care to elaborate?"
            ]
            
            positive_sentiment_responses = [
                "Great, so you liked \"{}\"! ",
                "Awesome, you enjoyed \"{}\"! ",
                "Nice! I see that you liked \"{}\". "
            ]
            
            negative_sentiment_responses = [
                "I see, you didn't enjoy \"{}\". ",
                "Got it, you didn't like \"{}\". ",
                "Understood, \"{}\" wasn't for you. "
            ]

            more_reviews_responses = [
                "Tell me about more movies so I can make recommendations.",
                "What other movies have you seen?",
                "Let me know about more movies you've watched!"
            ]

            recommendation_ready_responses = [
                "That's enough for me to make a recommendation!",
                "Great! I have enough info to suggest a movie for you.",
                "Awesome! I can now recommend something you might like."
            ]

            want_first_recommendation_responses = [
                "Would you like me to recommend you a movie? (yes/no)", 
                "Should I proceed with a recommendation? (yes/no)",
                "Are you up for a recommendation? (yes/no)"
            ]
            
            additional_recommendation_responses = [
                "I think you might enjoy \"{}\". ",
                "Based on your taste, you might like \"{}\"! ",
                "\"{}\" could be a great match for you! "
            ]

            want_another_recommendation_responses = [
                "Want another recommendation? (yes/no)", 
                "Would you like more recommendations? (yes/no)",
                "Interested in another recommendation? (yes/no)"
            ]
            
            no_more_recommendations_responses = [
                "I don't have any more recommendations for now! Let me know if you want to discuss more movies.",
                "That's all the recommendations I have at the moment! Feel free to tell me about more movies.",
                "No more suggestions from me for now! Let’s chat about more movies if you’d like."
            ]
            
            ending_recommendation_responses = [
                "Alright! Let me know if you want to talk about more movies.",
                "Got it! If you ever want more recommendations, just ask.",
                "No worries! I’m here whenever you want to discuss movies."
            ]
            
            if self.llm_enabled:
                return "I processed {} in LLM Programming mode!!".format(line)

            if self.awaiting_recommendation:
                line = re.sub(r"[^\w\s]", "", line)
                if line.lower() in affirmative_responses:
                    next_movie = self.recommendations.pop(0)
                    if not self.recommendations:
                        self.awaiting_recommendation = False
                    return random.choice(additional_recommendation_responses).format(self.titles[next_movie][0]) + '\n' + (random.choice(want_another_recommendation_responses) if self.recommendations else random.choice(no_more_recommendations_responses))
                elif line.lower() in negative_responses:
                    self.awaiting_recommendation = False
                    return random.choice(ending_recommendation_responses)
                else:
                    return "Please respond with 'yes' if you want another recommendation, or 'no' to stop."

            titles = self.extract_titles(line)
            if len(titles) == 0:
                return random.choice(unsure_responses)

            if len(titles) > 1:
                return random.choice(multiple_titles_responses)

            title = titles[0]
            movie_indices = self.find_movies_by_title(title)
            
            if not movie_indices:
                return random.choice(unknown_movie_responses).format(title)

            if len(movie_indices) > 1:
                return random.choice(multiple_matches_responses).format(title)

            sentiment = self.extract_sentiment(line)
            
            if sentiment == 0:
                return random.choice(neutral_sentiment_responses).format(title)
            
            idx = movie_indices[0]
            self.user_ratings[idx] = sentiment
            rated_movies = np.count_nonzero(self.user_ratings)

            if rated_movies >= 5:
                self.recommendations = self.recommend(self.user_ratings, self.ratings, k=5)
                if self.recommendations:
                    self.awaiting_recommendation = True
                    sentiment_response = random.choice(positive_sentiment_responses).format(title) if sentiment > 0 else random.choice(negative_sentiment_responses).format(title)
                    recommendation_ready_response = random.choice(recommendation_ready_responses)
                    return sentiment_response + '\n' + recommendation_ready_response + '\n' + random.choice(want_first_recommendation_responses)

            if sentiment > 0:
                return random.choice(positive_sentiment_responses).format(title) + '\n' + random.choice(more_reviews_responses)

            return random.choice(negative_sentiment_responses).format(title) + '\n' + random.choice(more_reviews_responses)



        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        # return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

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
        return re.findall(r'"(.*?)"', preprocessed_input)

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
        matching_indices = []
        title = title.strip()
        year_match = None
        year_extract = re.search(r'\((\d{4})\)$', title)
        if year_extract:
            year_match = year_extract.group(1)
            title = title.rsplit('(', 1)[0].strip()

        articles = ['The ', 'A ', 'An ']
        original_title = title
        real_article = ""
        for article in articles:
            if title.startswith(article):
                title = title[len(article):].strip()
                real_article += (", " + article)
                break

        

        title_variations = [
            original_title,
            title,
            (title + real_article).strip()
        ]
        for i, movie in enumerate(self.titles):
            movie_title = movie[0][:-7]
            movie_year = movie[0][-5:-1]
            

            for variation in title_variations:
                if year_match:
                    if variation == movie_title and year_match == movie_year:
                        matching_indices.append(i)
                        break

                else:
                    if variation == movie_title:
                        matching_indices.append(i)
                        break

        return matching_indices

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        stemmer = PorterStemmer()
        stemmed_dict = {stemmer.stem(key): value for key, value in self.sentiment.items()}
        words = preprocessed_input.lower().split()  # Tokenize and normalize
        negation_words = {'not', 'no', 'never', 'neither', 'nor', "didn't", "wouldn't"}
        emphasis_modifiers = {
            'very': 2,
            'extremely': 3,
            'incredibly': 3,
            'absolutely': 3,
            'totally': 2,
            'quite': 1.5,
            'somewhat': 0.5,
            'slightly': 0.5,
            'barely': 0.25,
            'kind of': 0.5,
            'sort of': 0.5
        }
    

        sentiment_score = 0
        negation_active = False
        in_quotes = False
        emphasis_multiplier = 1.0

        for word in words:
            if word.startswith('"') or word.endswith('"'):
                in_quotes = not in_quotes
                continue

            if in_quotes:
                continue
            
            if word in emphasis_modifiers:
                emphasis_multiplier = emphasis_modifiers[word]
                continue

            if word in negation_words:
                negation_active = not negation_active
                continue

            word = stemmer.stem(word)
            if word not in stemmed_dict: continue
            if stemmed_dict[word] == "pos":
                sentiment_score += emphasis_multiplier * (-1 if negation_active else 1)

            else:
                sentiment_score += emphasis_multiplier * (1 if negation_active else -1)

            emphasis_multiplier = 1.0

        if sentiment_score > 0:
            return 1
        elif sentiment_score < 0:
            return -1
        else:
            return 0

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
        binarized_ratings = np.zeros_like(ratings)  
        binarized_ratings[ratings > threshold] = 1  
        binarized_ratings[(ratings <= threshold) & (ratings > 0)] = -1
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

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


        u = np.ravel(u)
        v = np.ravel(v)

        dot_product = np.dot(u, v)

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        if norm_u == 0 or norm_v == 0:
            return 0.0 

        return dot_product / (norm_u * norm_v)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        # return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
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
        :param llm_enabled: whether the chatbot is in llm programming mode

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
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################


        num_movies, _ = ratings_matrix.shape

        norm_matrix = np.linalg.norm(ratings_matrix, axis=1, keepdims=True) 
        norm_matrix[norm_matrix == 0] = 1 
        similarity_matrix = (ratings_matrix @ ratings_matrix.T) / (norm_matrix @ norm_matrix.T)

        np.fill_diagonal(similarity_matrix, 0)

        unrated_indices = np.where(user_ratings == 0)[0]

        predicted_scores = similarity_matrix[unrated_indices] @ user_ratings

        top_k_indices = unrated_indices[np.argsort(predicted_scores)[::-1][:k]]

        return list(top_k_indices)


        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        # return recommendations

    ############################################################################
    # 4. PART 2: LLM Prompting Mode                                            #
    ############################################################################

    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is moviebot. You are a movie recommender chatbot. """ +\
        """You can help users find movies they like and provide information about movies."""

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt
    
    ############################################################################
    # 5. PART 3: LLM Programming Mode (also need to modify functions above!)   #
    ############################################################################

    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        return []

    ############################################################################
    # 6. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 7. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        This is a chatbot which gives movie recommendations. The user provides five movies 
        and their thoughts on said movies, and the chatbot will recommend them up to five
        movies which it thinks the user might like using sentiment analysis and a database of
        user reviews.
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
